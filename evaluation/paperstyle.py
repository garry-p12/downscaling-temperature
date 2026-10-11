"""Publication figure style: print sizes, vector output, panel labels.

WHY THIS IS SEPARATE FROM evaluation/mapping.py
    mapping.py styles SLIDE figures — 24 inches wide, 13 pt titles, a wrapped
    caption baked into the image so the sheet stands alone on a screen. None of
    that survives a journal. A paper figure is 3.5 or 7.2 inches wide, its
    caption lives in the LaTeX, its type has to stay legible after the
    publisher reduces it, and it has to be vector so the reviewer can zoom.
    Both styles are legitimate; they are for different readers.

THE RULES THIS ENCODES
    * Widths are the two standard column measures, so nothing is scaled by the
      publisher after the fact. A figure rescaled at submission has the wrong
      font size by definition.
    * Type is 8 pt base / 7 pt ticks, in a serif face with matching math, so a
      figure sits inside the text rather than beside it.
    * Every multi-panel figure gets (a), (b), (c) labels in the same place.
    * save() writes PDF and PNG together: the PDF goes in the manuscript, the
      PNG into slides and the review document.
    * The categorical colours are the Okabe-Ito set, which stays distinguishable
      under the common colour-vision deficiencies and in greyscale print.
"""
from __future__ import annotations

from pathlib import Path

# Standard journal measures, inches.
COL = 3.46          # single column (AGU/AMS ~88 mm)
WIDE = 7.20         # double column (~183 mm)
MID = 5.20          # 1.5 column, for a 2-panel row

# Okabe-Ito: colour-vision-deficiency safe, distinct in greyscale.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
YELLOW = "#E69F00"
SKY = "#56B4E9"
GREY = "#7F7F7F"
BLACK = "#1A1A1A"
CYCLE = [BLUE, ORANGE, GREEN, PURPLE, YELLOW, SKY]

# Map ramps, kept identical to evaluation/mapping.py so a paper figure and a
# slide figure of the same field are the same picture.
DIVERGING = ["#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948"]
SEQUENTIAL = ["#e8f1fd", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
NODATA = "#d9d9d9"


def use(serif: bool = True, base: float = 8.0):
    """Install the publication rcParams. Call once before creating figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler

    fam = (["Times New Roman", "STIXGeneral", "DejaVu Serif"] if serif
           else ["Helvetica", "Arial", "DejaVu Sans"])
    plt.rcParams.update({
        "font.family": "serif" if serif else "sans-serif",
        ("font.serif" if serif else "font.sans-serif"): fam,
        # STIX matches Times metrics, so inline maths does not jump out.
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "figure.titlesize": base + 1,

        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "axes.prop_cycle": cycler(color=CYCLE),

        # Ticks inward and on both used sides: the convention in most journals.
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,

        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,

        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderpad": 0.2,

        # White, not the slide surface: journals composite onto white paper.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Keep text as text in the PDF so it can be searched and restyled.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
    })


def panel(ax, letter: str, dx: float = -0.16, dy: float = 1.04, weight="bold"):
    """Put (a), (b), ... outside the top-left corner of an axes."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            ha="left", va="baseline", fontweight=weight, fontsize=9)


def ramp(stops):
    """A LinearSegmentedColormap from hex stops, with a defined no-data colour."""
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("r", stops)
    cm.set_bad(NODATA)
    return cm


def save(fig, out_dir, stem: str, dpi: int = 600, formats=("pdf", "png")):
    """Write the same figure as vector and raster, and report the print size."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    w, h = fig.get_size_inches()
    print(f"  {stem:<28} {w:.2f} x {h:.2f} in  "
          f"({w * 25.4:.0f} x {h * 25.4:.0f} mm)  -> "
          + ", ".join(p.suffix[1:] for p in paths))
    return paths


def graticule(ax, LAT, LON, step: float = 1.0, fontsize: float = 5.6):
    """True lat/lon contours and edge-exact ticks over an index-space imshow.

    The target grid is Albers, so lat/lon are curvilinear and an `extent=`
    bounding box would be a lie of up to a cell near the domain corners. The
    graticule is therefore drawn as contours of the actual coordinate arrays,
    and the ticks are placed where those contours meet the axes edge. Pass the
    SAME crop of LAT/LON as was passed to imshow.
    """
    import numpy as np

    ny, nx = LAT.shape
    lat_lv = np.arange(np.ceil(LAT.min() / step) * step, LAT.max(), step)
    lon_lv = np.arange(np.ceil(LON.min() / step) * step, LON.max(), step)
    for arr, lv in ((LON, lon_lv), (LAT, lat_lv)):
        if len(lv):
            ax.contour(arr, levels=lv, colors=GREY, linewidths=.3, alpha=.5,
                       linestyles=":")

    def edge(profile, levels, n):
        asc = profile[-1] > profile[0]
        xp = profile if asc else profile[::-1]
        fp = np.arange(n) if asc else np.arange(n)[::-1]
        t = [(np.interp(L, xp, fp), L) for L in levels
             if profile.min() <= L <= profile.max()]
        return [a for a, _ in t], [b for _, b in t]

    yt, yl = edge(LAT[:, 0], lat_lv, ny)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.0f}°N" for v in yl], fontsize=fontsize)
    xt, xl = edge(LON[-1, :], lon_lv, nx)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{abs(v):.0f}°W" for v in xl], fontsize=fontsize)
    ax.tick_params(length=1.8, pad=1.2)
    for sp in ax.spines.values():
        sp.set_visible(True)
