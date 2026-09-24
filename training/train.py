"""Train the downscaling model.

Temperature only: one head, one loss, one metric block.

Usage:
    python -m training.train                 # uses configs/*.yaml
    python -m training.train --epochs 2      # override for a quick smoke run
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import load_config
from models.model import build_model
from training.dataset import DownscaleDataset
from training.losses import decomposed_loss, temp_loss
from training.metrics import residual_spatial_corr, temperature_metrics
from training.tracking import Tracker
from training.viz import prediction_panel


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float,
                  min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def active_tasks(tcfg: dict) -> set[str]:
    """Heads to run. Kept as a set so build_model()'s contract stays uniform."""
    return {"temp"} if tcfg["tasks"]["temp"] else set()


def compute_loss(out: dict, y: torch.Tensor, ds: DownscaleDataset,
                 tasks: set[str], w: dict,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """Loss over cells that are BOTH land and outside the spatial holdout.

    The mask comes from the Dataset rather than being derived from the input's
    land_mask channel: it additionally zeroes the Austin holdout, which the
    input channel must NOT encode (telling the model those cells are ocean
    would corrupt its inputs, not just its gradient).
    """
    loss = y.new_zeros(())
    if "temp" in tasks and "temp" in out:
        ti = ds.target_index("tmp")
        # offset_weight is None -> the original single-term loss. Set it and the
        # loss splits into (spatial pattern) + offset_weight * (daily bias),
        # which is the README 5.4 decomposition made explicit.
        ow = w.get("offset_weight")
        if ow is None:
            loss = loss + w["temp"] * temp_loss(
                out["temp"], y[:, ti:ti + 1], w.get("ssim", 0.2),
                base=w.get("pixel_loss", "l1"), mask=mask)
        else:
            loss = loss + w["temp"] * decomposed_loss(
                out["temp"], y[:, ti:ti + 1], mask=mask,
                offset_weight=float(ow), ssim_weight=w.get("ssim", 0.2),
                base=w.get("pixel_loss", "l1"))
    return loss


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    return torch.device(name)


@torch.no_grad()
def validate(model, loader, ds, tasks, device) -> dict:
    """Validate in PHYSICAL units (°C) and against the bilinear baseline.

    Metrics are denormalized so val/rmse matches evaluate.py's units, and the
    (already target-gridded) coarse_tmp channel is scored as the bilinear
    baseline every epoch -> val/skill_rmse = baseline_rmse / model_rmse. >1
    means the model is beating interpolation; <=1 means it isn't earning its
    complexity yet. Makes each epoch's number falsifiable.
    """
    import numpy as np

    model.eval()
    nz = ds.nz
    ci = ds.in_names.index("coarse_tmp")
    ti = ds.target_index("tmp")
    preds, tgts, bilin, masks = [], [], [], []
    for x, y, m in loader:
        out = model(x.to(device), tasks)
        if "temp" not in out:
            continue
        preds.append(nz.inverse("out::tmp", out["temp"].cpu().numpy()))
        tgts.append(nz.inverse("out::tmp", y[:, ti:ti + 1].numpy()))
        bilin.append(nz.inverse("in::coarse_tmp", x[:, ci:ci + 1].numpy()))
        masks.append(m.numpy())
    if not preds:
        return {}
    pred, tgt, base = map(np.concatenate, (preds, tgts, bilin))
    # Anomalies -> absolute degC before scoring (see clim_for_samples).
    clim_t = ds.clim_for_samples("tmp")
    clim_c = ds.clim_for_samples("coarse_tmp")
    if clim_t is not None:
        n = pred.shape[0]
        pred = pred + clim_t[:n, None]
        tgt = tgt + clim_t[:n, None]
        if clim_c is not None:
            base = base + clim_c[:n, None]
    # Score LAND ONLY. Ocean cells are zero-filled, and including them would
    # quietly flatter every metric (a perfect 0 == 0 match over water).
    if masks:
        m = np.concatenate(masks) > 0.5
        pred, tgt, base = (np.where(m, a, np.nan) for a in (pred, tgt, base))
    m = temperature_metrics(pred, tgt)                     # physical °C
    base_rmse = temperature_metrics(base, tgt)["rmse"]
    m["bilinear_rmse"] = base_rmse
    m["skill_rmse"] = base_rmse / (m["rmse"] + 1e-9)       # >1 == beats bilinear
    # Fine-scale skill after removing the bilinear floor (raw corr ~0.99 is a
    # floor, this isn't). Model corr vs baseline corr (=0 by construction).
    m["resid_corr"] = residual_spatial_corr(pred, tgt, base)
    return m


def main(overrides: dict) -> None:
    mcfg = load_config("model")
    tcfg = load_config("train")
    tcfg.update({k: v for k, v in overrides.items()
                 if v is not None and k != "arch"})
    if overrides.get("arch"):
        mcfg["arch"] = overrides["arch"]

    # --seed overrides the config so the SAME architecture can be trained
    # repeatedly under different initializations; without repeats a 0.005 degC
    # gap between models cannot be distinguished from run-to-run noise.
    if overrides.get("seed") is not None:
        tcfg["seed"] = int(overrides["seed"])
    device = resolve_device(tcfg["device"])
    # Printed because getting this wrong is silent: bf16 autocast only engages on
    # CUDA, so a laptop run and a GPU run of identical code differ in numerics.
    print(f"[train] device={device} precision={tcfg['precision']} "
          f"seed={tcfg['seed']}")
    torch.manual_seed(tcfg["seed"])

    # --zarr wins over configs/train.yaml so a smoke run cannot be pointed at
    # (and cannot overwrite) a real multi-gigabyte store by forgetting to edit
    # the config back.
    zarr = overrides.get("zarr") or tcfg["data"]["zarr"]
    norm = str(Path(zarr).parent / "norm_stats.json")
    patch = tcfg["data"]["patch_size"]
    if overrides.get("patch_size") is not None:
        # 0 means "no cropping": the decomposed loss needs the whole field to
        # compute a domain mean, and a 48x48 patch mean is a poor stand-in for
        # it (proxy error sd 0.376 against an offset sd of 0.463).
        patch = None if int(overrides["patch_size"]) <= 0 \
            else int(overrides["patch_size"])
    # Spatial holdout: training and validation patches both avoid the
    # evaluation region, so model SELECTION never sees it either.
    dcfg = load_config("data")
    holdout = dcfg.get("holdout")
    ppS = tcfg["data"].get("patches_per_sample", 1)
    if patch is None and ppS > 1:
        # Drawing N "patches" from an uncropped field yields N identical copies
        # of the same day — N times the compute for no new information.
        print(f"[train] patch_size=None, so patches_per_sample {ppS} -> 1")
        ppS = 1
    # Covariate ablations select a subset of the superset store's channels.
    use_ch = overrides.get("use_channels") or dcfg["dataset"].get("use_channels")
    train_ds = DownscaleDataset(zarr, norm, "train", patch, tcfg["seed"],
                                holdout=holdout,
                                holdout_mode="exclude" if holdout else None,
                                patches_per_sample=ppS, use_channels=use_ch)
    # patch=None: validate on whole fields. Patch-cropping validation would
    # break the anomaly->degC conversion (climatology is full-grid) and make
    # val metrics depend on which crops were drawn.
    val_ds = DownscaleDataset(zarr, norm, "val", None, tcfg["seed"] + 1,
                              holdout=holdout,
                              holdout_mode="exclude" if holdout else None,
                              use_channels=use_ch)
    # Workers are respawned every epoch and each one opens the store. On a small
    # local Zarr that churn can dominate when the machine is already under
    # memory pressure, so allow 0 (load in-process). Default stays the config
    # value so cluster runs are unaffected.
    n_workers = tcfg["data"]["num_workers"] if overrides.get("num_workers") is None \
        else int(overrides["num_workers"])
    train_ld = DataLoader(train_ds, batch_size=tcfg["data"]["batch_size"],
                          shuffle=True, num_workers=n_workers,
                          drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Derive the input width from the DATA, not configs/model.yaml. An ablation
    # changes the channel count, and a mismatch here would either crash late or
    # silently train against the wrong first-layer shape. Recording the channel
    # NAMES in the checkpoint makes it self-describing, so evaluation rebuilds
    # the same input stack without being told which variant this was.
    mcfg["in_channels"] = len(train_ds.in_names)
    mcfg["use_channels"] = list(train_ds.in_names)
    arch = mcfg.get("arch", "swin")
    model = build_model(mcfg).to(device)
    print(f"[train] {len(train_ds.in_names)} input channels: "
          f"{', '.join(train_ds.in_names)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] arch={arch} params={n_params/1e6:.2f}M")

    wandb_cfg = dict(tcfg["logging"].get("wandb", {}))
    if wandb_cfg.get("run_name") is None:
        wandb_cfg["run_name"] = overrides.get("run_name") or arch
    tracker = Tracker(wandb_cfg, {"model": dict(mcfg), "train": dict(tcfg)},
                      {"params_M": n_params / 1e6, "device": str(device),
                       "arch": arch})
    tracker.watch(model, wandb_cfg)
    if tracker.enabled:
        print(f"[train] wandb: {tracker.run.url}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["optim"]["lr"],
                            weight_decay=tcfg["optim"]["weight_decay"],
                            betas=tuple(tcfg["optim"]["betas"]))
    epochs = int(overrides.get("epochs") or tcfg["schedule"]["epochs"])
    steps_per = max(1, len(train_ld))
    total_steps = epochs * steps_per
    warmup = tcfg["schedule"]["warmup_epochs"] * steps_per

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tcfg["precision"])
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg["precision"] == "fp16" and use_amp))

    tasks = active_tasks(tcfg)
    # Loss-term overrides live here so an ablation is a CLI flag rather than a
    # config edit — the ssim weight is reported as a metric AND optimized, so
    # its contribution has to be separable.
    w = dict(tcfg["loss_weights"])
    if overrides.get("ssim_weight") is not None:
        w["ssim"] = float(overrides["ssim_weight"])
    if overrides.get("pixel_loss"):
        w["pixel_loss"] = overrides["pixel_loss"]
    if overrides.get("offset_weight") is not None:
        w["offset_weight"] = float(overrides["offset_weight"])
        if patch is not None:
            raise SystemExit(
                f"--offset-weight needs full fields but patch_size={patch}. A "
                f"patch mean is a poor proxy for the domain offset (proxy error "
                f"sd 0.376 vs offset sd 0.463), so the split would remove real "
                f"spatial structure instead of the bias. Pass --patch-size 0.")
    if w.get("offset_weight") is None:
        print(f"[train] loss: {w.get('pixel_loss','l1')} + "
              f"{w['ssim']}*(1-SSIM)")
    else:
        print(f"[train] loss: DECOMPOSED — {w.get('pixel_loss','l1')}(de-meaned)"
              f" + {w['ssim']}*(1-SSIM) + {w['offset_weight']}*|daily offset|")
    # Per-arch subdirectory: comparison runs must not overwrite each other's
    # best.pt. 'swin' keeps the flat path so existing checkpoints stay valid.
    # run_name keeps seed repeats in separate directories; ckpt_path() resolves
    # "<arch>_s<seed>" the same way it resolves a bare arch name.
    run_name = overrides.get("run_name")
    ckpt_dir = Path(tcfg["logging"]["ckpt_dir"])
    if run_name:
        ckpt_dir = ckpt_dir / run_name
    elif arch != "swin":
        ckpt_dir = ckpt_dir / arch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] tasks={sorted(tasks)} epochs={epochs} ckpt_dir={ckpt_dir}")

    step = 0
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for x, y, m in train_ld:
            lr = cosine_warmup(step, warmup, total_steps, tcfg["optim"]["lr"],
                               tcfg["schedule"]["min_lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                out = model(x, tasks)
                loss = compute_loss(out, y, train_ds, tasks, w, m.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % tcfg["logging"]["log_every"] == 0:
                print(f"[train] e{epoch} s{step} lr={lr:.2e} loss={loss.item():.4f}")
                tracker.log({"train/loss": loss.item(), "train/lr": lr,
                             "epoch": epoch}, step=step)
            step += 1

        if (epoch + 1) % tcfg["logging"]["val_every"] == 0:
            m = validate(model, val_ld, val_ds, tasks, device)
            print(f"[val] e{epoch} {m}")
            tracker.log({f"val/{k}": v for k, v in m.items()} | {"epoch": epoch},
                        step=step)
            if tracker.enabled and wandb_cfg.get("log_images", False):
                n_img = wandb_cfg.get("num_images", 4)
                for field in val_ds.out_names:   # temp-only build => just 'tmp'
                    fig = prediction_panel(model, val_ds, tasks, device, n_img, field)
                    if fig is not None:
                        tracker.log_figure(f"pred/{field}", fig, step=step)
            score = m.get("rmse", loss.item())
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "cfg": dict(mcfg)}, ckpt_dir / "best.pt")
                print(f"[train] saved best (rmse={score:.4f})")
                tracker.summary("best/val_rmse", best)
    torch.save({"model": model.state_dict(), "epoch": epoch,
                "cfg": dict(mcfg)}, ckpt_dir / "last.pt")
    tracker.finish()
    print("[train] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="override configs/train.yaml seed (for repeats)")
    ap.add_argument("--run-name", default=None,
                    help="checkpoint subdirectory, e.g. deepsd_s1")
    ap.add_argument("--ssim-weight", type=float, default=None,
                    help="override loss_weights.ssim (0.0 = pure pixel loss)")
    ap.add_argument("--offset-weight", type=float, default=None,
                    help="split the loss into spatial pattern + this weight "
                         "times the uniform daily offset (README 5.4). Needs "
                         "--patch-size 0: a patch mean is a poor proxy for the "
                         "domain offset. Unset = the original single-term loss.")
    ap.add_argument("--patch-size", type=int, default=None,
                    help="override data.patch_size; 0 = train on full fields")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override data.num_workers (0 = load in-process)")
    ap.add_argument("--pixel-loss", default=None, choices=["l1", "mse", "amse"],
                    help="override loss_weights.pixel_loss")
    ap.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"],
                    help="override configs/train.yaml precision. USE fp32 ON GPU: "
                         "bf16's 8-bit mantissa cannot hold the SSIM term's local "
                         "variance/covariance products and roughly halves SSIM "
                         "(README.md section 9)")
    ap.add_argument("--use-channels", nargs="+", default=None,
                    help="input channels to select from the store, in order "
                         "(covariate ablations); default = all stored channels")
    ap.add_argument("--zarr", default=None,
                    help="override configs/train.yaml data.zarr (e.g. a "
                         "synthetic store built by data.build_dataset)")
    from training.run_all import ALL_ARCHS  # one list, no drift

    ap.add_argument("--arch", default=None, choices=ALL_ARCHS,
                    help="override configs/model.yaml arch for this run")
    args = ap.parse_args()
    main({"epochs": args.epochs, "arch": args.arch, "zarr": args.zarr,
          "use_channels": args.use_channels, "precision": args.precision,
          "seed": args.seed, "run_name": args.run_name,
          "ssim_weight": args.ssim_weight, "pixel_loss": args.pixel_loss,
          "num_workers": args.num_workers,
          "offset_weight": args.offset_weight, "patch_size": args.patch_size})
