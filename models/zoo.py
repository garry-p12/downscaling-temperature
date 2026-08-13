"""Transformer (and modern-conv) architectures for the downscaling comparison.

The point is to cover the *design space* of attention, not to collect many
near-duplicates. Each entry here uses a structurally different mechanism:

    swinir     local windowed attention, FLAT (no hierarchy)
    restormer  channel / transposed attention (MDTA) — cost is independent of
               spatial size, which matters on a 49x40 grid
    segformer  global attention with spatial-reduction keys, hierarchical,
               all-MLP decoder
    maxvit     block attention (local) + grid attention (sparse global),
               interleaved
    convnext   modern conv U-Net — the control that isolates "attention" from
               "everything else modern architectures do"

Together with the existing swin (hierarchical windowed attention), unet, and
deepsd, that is eight methods spanning local / channel / global / sparse
attention plus two conv references.

Every model obeys the shared contract:
    forward(x: (B, C_in, H, W), tasks=None) -> {"temp": (B, 1, H, W)}
with H, W preserved exactly. Odd grids are real (the Colorado domain is 49x40),
so each model pads internally to its own size requirement and crops back.

All are implemented from scratch — no pretrained weights, no external
dependencies. Sizes are tuned for a small regional grid, not ImageNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_swin import DropPath, SwinStage


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
class SizeGuard:
    """Mixin: pad to a multiple of ``divisor`` on entry, crop on exit."""

    divisor: int = 1

    def _pad(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.divisor, (-W) % self.divisor
        if ph or pw:
            mode = "reflect" if (ph < H and pw < W) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return x, H, W

    @staticmethod
    def _crop(x, H, W):
        return x[..., :H, :W]


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an (B, C, H, W) tensor."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- #
# 1. SwinIR — flat windowed attention
# --------------------------------------------------------------------------- #
class SwinIR(nn.Module, SizeGuard):
    """Windowed attention at ONE resolution, with a global residual.

    The contrast against the hierarchical Swin U-Net is the point: on a 49x40
    domain the U-Net variant downsamples internally to 3x2, which may destroy
    more than the multi-scale context is worth. This keeps full resolution
    throughout.
    """

    def __init__(self, in_channels: int, dim: int = 96, depths=(6, 6, 6, 6),
                 num_heads: int = 6, window_size: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.divisor = window_size
        self.head_conv = nn.Conv2d(in_channels, dim, 3, padding=1,
                                   padding_mode="reflect")
        self.groups = nn.ModuleList([
            SwinStage(dim, d, num_heads, window_size, mlp_ratio, [0.0] * d)
            for d in depths
        ])
        self.group_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect")
            for _ in depths
        ])
        self.norm = nn.LayerNorm(dim)
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(dim, 1, 3, padding=1, padding_mode="reflect"))

    def forward(self, x, tasks=None):
        del tasks
        x, H0, W0 = self._pad(x)
        feat = self.head_conv(x)
        shallow = feat
        B, C, H, W = feat.shape
        for stage, conv in zip(self.groups, self.group_convs):
            tok = feat.flatten(2).transpose(1, 2)
            tok = stage(tok, H, W)
            tok = self.norm(tok)
            feat = feat + conv(tok.transpose(1, 2).view(B, C, H, W))
        return {"temp": self._crop(self.tail(feat + shallow), H0, W0)}


# --------------------------------------------------------------------------- #
# 2. Restormer — channel (transposed) attention
# --------------------------------------------------------------------------- #
class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention: attention across CHANNELS.

    The Gram matrix is (C x C), not (HW x HW), so cost does not grow with
    spatial size. On a 49x40 grid that is a structural advantage — there are
    only 1960 spatial positions but plenty of channel structure to model.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3,
                                padding_mode="reflect")
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        r = lambda t: t.view(B, self.num_heads, C // self.num_heads, H * W)  # noqa: E731
        q, k, v = map(r, (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn.softmax(-1) @ v).reshape(B, C, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv feed-forward network."""

    def __init__(self, dim: int, expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                            groups=hidden * 2, padding_mode="reflect")
        self.project_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        a, b = self.dw(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.n1, self.attn = LayerNorm2d(dim), MDTA(dim, num_heads)
        self.n2, self.ffn = LayerNorm2d(dim), GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Restormer(nn.Module, SizeGuard):
    def __init__(self, in_channels: int, dim: int = 48, depths=(4, 6, 6, 8),
                 heads=(1, 2, 4, 8)):
        super().__init__()
        self.divisor = 2 ** (len(depths) - 1)
        self.embed = nn.Conv2d(in_channels, dim, 3, padding=1,
                               padding_mode="reflect")
        dims = [dim * (2 ** i) for i in range(len(depths))]

        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        for i, d in enumerate(depths):
            self.enc.append(nn.Sequential(
                *[RestormerBlock(dims[i], heads[i]) for _ in range(d)]))
            self.down.append(
                nn.Sequential(nn.Conv2d(dims[i], dims[i] * 2, 3, stride=2,
                                        padding=1, padding_mode="reflect"))
                if i < len(depths) - 1 else None)

        self.up, self.fuse, self.dec = (nn.ModuleList() for _ in range(3))
        for i in range(len(depths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, stride=2))
            self.fuse.append(nn.Conv2d(2 * dims[i - 1], dims[i - 1], 1))
            self.dec.append(nn.Sequential(
                *[RestormerBlock(dims[i - 1], heads[i - 1])
                  for _ in range(depths[i - 1])]))
        self.tail = nn.Conv2d(dim, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x, tasks=None):
        del tasks
        x, H0, W0 = self._pad(x)
        feat = self.embed(x)
        skips = []
        for i, stage in enumerate(self.enc):
            feat = stage(feat)
            if self.down[i] is not None:
                skips.append(feat)
                feat = self.down[i](feat)
        for up, fuse, dec, skip in zip(self.up, self.fuse, self.dec,
                                       reversed(skips)):
            feat = up(feat)
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode="nearest")
            feat = dec(fuse(torch.cat([feat, skip], dim=1)))
        return {"temp": self._crop(self.tail(feat), H0, W0)}


# --------------------------------------------------------------------------- #
# 3. SegFormer / MiT — global attention with spatial-reduction keys
# --------------------------------------------------------------------------- #
class EfficientAttention(nn.Module):
    """Global self-attention with keys/values spatially reduced by ``sr``.

    Every query attends everywhere (unlike windowed attention), but keys are
    pooled, so cost is O(HW * HW/sr^2) instead of O((HW)^2).
    """

    def __init__(self, dim: int, num_heads: int, sr: int = 1):
        super().__init__()
        self.num_heads, self.sr = num_heads, sr
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr > 1:
            self.reduce = nn.Conv2d(dim, dim, sr, stride=sr)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        if self.sr > 1:
            y = x.transpose(1, 2).view(B, C, H, W)
            y = self.reduce(y).flatten(2).transpose(1, 2)
            y = self.norm(y)
        else:
            y = x
        kv = self.kv(y).view(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MixFFN(nn.Module):
    """FFN with a depthwise conv — SegFormer's replacement for position embeds."""

    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden,
                            padding_mode="reflect")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        x = self.fc1(x)
        x = self.dw(x.transpose(1, 2).view(B, -1, H, W)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, sr, drop_path=0.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), EfficientAttention(dim, num_heads, sr)
        self.n2, self.ffn = nn.LayerNorm(dim), MixFFN(dim)
        self.dp = DropPath(drop_path)

    def forward(self, x, H, W):
        x = x + self.dp(self.attn(self.n1(x), H, W))
        return x + self.dp(self.ffn(self.n2(x), H, W))


class SegFormer(nn.Module, SizeGuard):
    def __init__(self, in_channels: int, dims=(32, 64, 160, 256),
                 depths=(2, 2, 2, 2), heads=(1, 2, 5, 8), srs=(4, 2, 1, 1),
                 decoder_dim: int = 256):
        super().__init__()
        self.divisor = 8
        strides = (1, 2, 2, 2)          # stage 1 keeps full res (small domain)
        self.patch_embeds = nn.ModuleList()
        cin = in_channels
        for d, s in zip(dims, strides):
            self.patch_embeds.append(
                nn.Conv2d(cin, d, 3 if s == 1 else 3, stride=s, padding=1,
                          padding_mode="reflect"))
            cin = d
        self.stages = nn.ModuleList([
            nn.ModuleList([MiTBlock(dims[i], heads[i], srs[i])
                           for _ in range(depths[i])])
            for i in range(len(dims))])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.linears = nn.ModuleList([nn.Conv2d(d, decoder_dim, 1) for d in dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(dims), decoder_dim, 1),
            nn.GELU(), LayerNorm2d(decoder_dim))
        self.tail = nn.Conv2d(decoder_dim, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x, tasks=None):
        del tasks
        x, H0, W0 = self._pad(x)
        outs = []
        for embed, blocks, norm in zip(self.patch_embeds, self.stages, self.norms):
            x = embed(x)
            B, C, H, W = x.shape
            tok = x.flatten(2).transpose(1, 2)
            for blk in blocks:
                tok = blk(tok, H, W)
            x = norm(tok).transpose(1, 2).view(B, C, H, W)
            outs.append(x)
        target = outs[0].shape[-2:]
        fused = self.fuse(torch.cat(
            [F.interpolate(lin(o), size=target, mode="bilinear",
                           align_corners=False)
             for lin, o in zip(self.linears, outs)], dim=1))
        out = self.tail(fused)
        if out.shape[-2:] != (H0, W0):
            out = F.interpolate(out, size=(H0, W0), mode="bilinear",
                                align_corners=False)
        return {"temp": out}


# --------------------------------------------------------------------------- #
# 4. MaxViT — block (local) + grid (sparse global) attention
# --------------------------------------------------------------------------- #
class RelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.h = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                       # x: (B*, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class MaxViTBlock(nn.Module):
    """MBConv -> block attention (within p x p tiles) -> grid attention
    (across tiles, i.e. a strided/dilated view). Local plus sparse-global in
    one block, at linear cost."""

    def __init__(self, dim: int, num_heads: int, p: int = 7):
        super().__init__()
        self.p = p
        self.mb = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim * 2, 1), nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1))
        self.nb, self.block_attn = nn.LayerNorm(dim), RelAttention(dim, num_heads)
        self.ng, self.grid_attn = nn.LayerNorm(dim), RelAttention(dim, num_heads)
        self.nf = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.mb(x)
        p = self.p
        ph, pw = (-H) % p, (-W) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        Hp, Wp = x.shape[-2:]
        gh, gw = Hp // p, Wp // p

        # Block attention: attend within each p x p tile.
        y = x.view(B, C, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1)
        y = y.reshape(-1, p * p, C)
        y = y + self.block_attn(self.nb(y))
        y = y.view(B, gh, gw, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

        # Grid attention: attend across tiles at matching offsets (sparse global).
        z = y.view(B, C, gh, p, gw, p).permute(0, 3, 5, 2, 4, 1)
        z = z.reshape(-1, gh * gw, C)
        z = z + self.grid_attn(self.ng(z))
        z = z.view(B, p, p, gh, gw, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, Hp, Wp)

        out = z[..., :H, :W]
        tok = out.flatten(2).transpose(1, 2)
        tok = tok + self.ffn(self.nf(tok))
        return tok.transpose(1, 2).view(B, C, H, W)


class MaxViT(nn.Module, SizeGuard):
    def __init__(self, in_channels: int, dim: int = 96, depth: int = 6,
                 num_heads: int = 4, p: int = 7):
        super().__init__()
        self.divisor = 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect"))
        self.blocks = nn.ModuleList(
            [MaxViTBlock(dim, num_heads, p) for _ in range(depth)])
        self.tail = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(dim, 1, 3, padding=1, padding_mode="reflect"))

    def forward(self, x, tasks=None):
        del tasks
        feat = self.stem(x)
        for blk in self.blocks:
            feat = blk(feat)
        return {"temp": self.tail(feat)}


# --------------------------------------------------------------------------- #
# 5. Plain ViT — isotropic, full global attention (SETR/DPT-lite decoder)
# --------------------------------------------------------------------------- #
class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), RelAttention(dim, num_heads)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        self.dp = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp(self.attn(self.n1(x)))
        return x + self.dp(self.mlp(self.n2(x)))


class ViT(nn.Module, SizeGuard):
    """The canonical vision transformer: patch-embed, add position embeddings,
    then N blocks of FULL global self-attention at a single scale.

    No windowing, no hierarchy, no locality prior of any kind — every patch
    attends to every other patch. On this domain a patch_size of 4 gives about
    13x10 = 130 tokens, so full quadratic attention is entirely affordable;
    the question is whether the total absence of an inductive bias hurts when
    training data is ~1100 days.

    Position embeddings are interpolated to whatever token grid arrives, so the
    model still works if the domain size changes.
    """

    def __init__(self, in_channels: int, dim: int = 384, depth: int = 8,
                 num_heads: int = 6, patch_size: int = 4,
                 grid: tuple = (16, 16)):
        super().__init__()
        self.patch_size = patch_size
        self.divisor = patch_size
        self.embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, dim, *grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # Progressive upsampling decoder back to full resolution.
        ups, cur = [], dim
        f = patch_size
        while f > 1:
            ups += [nn.Conv2d(cur, cur // 2, 3, padding=1,
                              padding_mode="reflect"),
                    nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear",
                                           align_corners=False)]
            cur //= 2
            f //= 2
        self.decoder = nn.Sequential(*ups)
        self.tail = nn.Conv2d(cur, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x, tasks=None):
        del tasks
        x, H0, W0 = self._pad(x)
        feat = self.embed(x)
        B, C, H, W = feat.shape
        pos = F.interpolate(self.pos, size=(H, W), mode="bicubic",
                            align_corners=False)
        tok = (feat + pos).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tok = blk(tok)
        feat = self.norm(tok).transpose(1, 2).view(B, C, H, W)
        out = self.tail(self.decoder(feat))
        if out.shape[-2:] != (H0, W0):
            out = F.interpolate(out, size=(H0, W0), mode="bilinear",
                                align_corners=False)
        return {"temp": out}


# --------------------------------------------------------------------------- #
# 6. ConvNeXt U-Net — the "is it attention, or just modern design?" control
# --------------------------------------------------------------------------- #
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        # Zero padding, not reflect: at the deepest stage of a 49x40 domain the
        # feature map is ~6x5 (or 4x2 on smaller crops), and reflect padding
        # requires pad < dim, so a 7x7 kernel would raise. ConvNeXt uses zero
        # padding in the original anyway.
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1, self.pw2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.dp = DropPath(drop_path)

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.norm(y)))) * self.gamma
        return x + self.dp(y.permute(0, 3, 1, 2))


class ConvNeXtUNet(nn.Module, SizeGuard):
    def __init__(self, in_channels: int, base: int = 96, depths=(2, 2, 4, 2)):
        super().__init__()
        self.divisor = 2 ** (len(depths) - 1)
        dims = [base * (2 ** i) for i in range(len(depths))]
        self.stem = nn.Conv2d(in_channels, base, 3, padding=1,
                              padding_mode="reflect")
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        for i, d in enumerate(depths):
            self.enc.append(nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(d)]))
            self.down.append(
                nn.Sequential(LayerNorm2d(dims[i]),
                              nn.Conv2d(dims[i], dims[i] * 2, 2, stride=2))
                if i < len(depths) - 1 else None)
        self.up, self.fuse, self.dec = (nn.ModuleList() for _ in range(3))
        for i in range(len(depths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, stride=2))
            self.fuse.append(nn.Conv2d(2 * dims[i - 1], dims[i - 1], 1))
            self.dec.append(nn.Sequential(
                *[ConvNeXtBlock(dims[i - 1]) for _ in range(depths[i - 1])]))
        self.tail = nn.Conv2d(base, 1, 3, padding=1, padding_mode="reflect")

    def forward(self, x, tasks=None):
        del tasks
        x, H0, W0 = self._pad(x)
        feat = self.stem(x)
        skips = []
        for i, stage in enumerate(self.enc):
            feat = stage(feat)
            if self.down[i] is not None:
                skips.append(feat)
                feat = self.down[i](feat)
        for up, fuse, dec, skip in zip(self.up, self.fuse, self.dec,
                                       reversed(skips)):
            feat = up(feat)
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode="nearest")
            feat = dec(fuse(torch.cat([feat, skip], dim=1)))
        return {"temp": self._crop(self.tail(feat), H0, W0)}


# --------------------------------------------------------------------------- #
# 7. EDSR — the strong residual-CNN SR baseline
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """EDSR residual block: conv-ReLU-conv, no normalization, scaled residual.

    Removing BatchNorm is EDSR's central claim — normalization layers discard
    range information that super-resolution needs. The residual scaling keeps
    deep stacks stable without it.
    """

    def __init__(self, dim: int, res_scale: float = 0.1):
        super().__init__()
        pm = "reflect"
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode=pm), nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode=pm))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class EDSR(nn.Module, SizeGuard):
    """Deep residual CNN. The baseline a transformer must actually beat —
    climate-downscaling papers repeatedly find well-tuned CNNs close most of
    the gap, so this is the honest reference point, not a strawman."""

    def __init__(self, in_channels: int, dim: int = 128, n_blocks: int = 16,
                 res_scale: float = 0.1):
        super().__init__()
        self.divisor = 1
        pm = "reflect"
        self.head = nn.Conv2d(in_channels, dim, 3, padding=1, padding_mode=pm)
        self.body = nn.Sequential(
            *[ResBlock(dim, res_scale) for _ in range(n_blocks)],
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode=pm))
        self.tail = nn.Conv2d(dim, 1, 3, padding=1, padding_mode=pm)

    def forward(self, x, tasks=None):
        del tasks
        f = self.head(x)
        f = f + self.body(f)                    # long skip over the whole body
        return {"temp": self.tail(f)}


# --------------------------------------------------------------------------- #
# 8. ESRT — efficient hybrid CNN + transformer, built for small datasets
# --------------------------------------------------------------------------- #
class ReductionAttention(nn.Module):
    """Self-attention on a spatially reduced token grid, upsampled back.

    ESRT's efficiency trick: attend on a coarse token map rather than every
    pixel. On a 48x48 patch that is 12x12 = 144 tokens instead of 2304, which
    is what makes a transformer trainable on ~1100 days of data without
    overfitting.
    """

    def __init__(self, dim: int, num_heads: int, reduction: int = 4):
        super().__init__()
        self.r = reduction
        self.norm = LayerNorm2d(dim)
        self.attn = RelAttention(dim, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.norm(x)
        # avg_pool2d(ceil_mode=True), not adaptive_avg_pool2d: MPS requires the
        # input size to be divisible by the output size for adaptive pooling,
        # and real grids are ragged (the Austin crop is 35x31). ceil_mode keeps
        # the partial edge window instead of dropping it.
        small = F.avg_pool2d(y, kernel_size=self.r, stride=self.r,
                             ceil_mode=True)
        h, w = small.shape[-2:]
        tok = small.flatten(2).transpose(1, 2)
        tok = self.attn(tok)
        out = tok.transpose(1, 2).view(B, C, h, w)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return x + out


class ESRTBlock(nn.Module):
    """Lightweight conv feature extraction + reduced-token attention."""

    def __init__(self, dim: int, num_heads: int, reduction: int):
        super().__init__()
        pm = "reflect"
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode=pm), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode=pm))
        self.attn = ReductionAttention(dim, num_heads, reduction)
        self.ffn = nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, dim * 2, 1),
                                 nn.GELU(), nn.Conv2d(dim * 2, dim, 1))

    def forward(self, x):
        x = x + self.conv(x)
        x = self.attn(x)
        return x + self.ffn(x)


class ESRT(nn.Module, SizeGuard):
    def __init__(self, in_channels: int, dim: int = 64, depth: int = 6,
                 num_heads: int = 4, reduction: int = 4):
        super().__init__()
        self.divisor = 1
        pm = "reflect"
        self.head = nn.Conv2d(in_channels, dim, 3, padding=1, padding_mode=pm)
        self.blocks = nn.ModuleList(
            [ESRTBlock(dim, num_heads, reduction) for _ in range(depth)])
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode=pm), nn.GELU(),
            nn.Conv2d(dim, 1, 3, padding=1, padding_mode=pm))

    def forward(self, x, tasks=None):
        del tasks
        f = self.head(x)
        shallow = f
        for blk in self.blocks:
            f = blk(f)
        return {"temp": self.tail(f + shallow)}
