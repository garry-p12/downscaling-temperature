"""Shared map furniture: the graticule, the palettes, the holdout box.

Extracted so every figure in the repo draws the SAME geography. The grid is
EPSG:5070 Albers, so lines of constant latitude CURVE across the image — one
pixel row spans ~0.3 deg of latitude. Ticks are therefore only exact on the
left and bottom edges, where they are computed; away from those edges the
dotted contours are the truth. An earlier version of this code kept two
separate lists for the contour levels and the tick labels, and they silently
disagreed by a full degree (~110 km). One definition now serves both.
"""
from __future__ import annotations

import numpy as np

LAT_LEVELS = np.arange(28, 35, 2)        # 28 .. 34 N
LON_LEVELS = np.arange(-104, -93, 2)     # 104 .. 94 W

SURF, INK, INK2, INK3 = "#fcfdfe", "#0f1419", "#55606e", "#7d8996"
ORANGE, GRAT, NODATA = "#eb6834", "#5b6b7a", "#c9c6bf"


def palettes():
    """(sequential, diverging) colormaps with a defined 'no data' colour."""
    from matplotlib.colors import LinearSegmentedColormap

    seq = LinearSegmentedColormap.from_list(
        "s", ["#e8f1fd", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    div = LinearSegmentedColormap.from_list(
        "d", ["#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948"])
    for c in (seq, div):
        c.set_bad(NODATA)
    return seq, div


def graticule(ax, LAT, LON, fontsize: float = 7.0) -> None:
    """True lat/lon contours plus edge-exact ticks."""
    ny, nx = LAT.shape
    ax.contour(LON, levels=LON_LEVELS, colors=GRAT, linewidths=.5,
               alpha=.45, linestyles=":")
    ax.contour(LAT, levels=LAT_LEVELS, colors=GRAT, linewidths=.5,
               alpha=.45, linestyles=":")

    def edge(profile, levels, n):
        asc = profile[-1] > profile[0]
        xp = profile if asc else profile[::-1]
        fp = np.arange(n) if asc else np.arange(n)[::-1]
        t = [(np.interp(L, xp, fp), L) for L in levels
             if profile.min() <= L <= profile.max()]
        return [a for a, _ in t], [b for _, b in t]

    yt, yl = edge(LAT[:, 0], LAT_LEVELS, ny)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.0f}N" for v in yl], fontsize=fontsize, color=INK3)
    xt, xl = edge(LON[-1, :], LON_LEVELS, nx)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{abs(v):.0f}W" for v in xl], fontsize=fontsize, color=INK3)
    ax.tick_params(length=2.5, color=GRAT, pad=1.5)


def holdout_box(ax, box, label: str = "HELD-OUT REGION",
                fontsize: float = 6.5) -> None:
    """Outline the evaluated region. Not 'Austin': it is the Austin box plus a
    1 deg buffer, 370x350 km, and it also contains San Antonio, Waco, Killeen
    and College Station."""
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    i0, i1, j0, j1 = box
    ax.add_patch(plt.Rectangle((j0 - .5, i0 - .5), j1 - j0, i1 - i0,
                               fill=False, ec=ORANGE, lw=1.5, zorder=6))
    if label:
        ax.text((j0 + j1) / 2, i0 - 1.6, label, ha="center", va="bottom",
                fontsize=fontsize, color=ORANGE, fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=SURF)])


def header(fig, title: str, body: str, chars: int = 200,
           y_title: float = .995, fontsize: float = 9.3,
           linespacing: float = 1.45):
    """Figure title plus a hard-wrapped caption, returning the rect top.

    matplotlib does not wrap fig.text, and savefig(bbox_inches='tight') grows
    the canvas to contain the overhang — a long single-line caption silently
    stretched a 19.5x9.8 in figure to a 49-inch-wide image with unreadable
    maps. Wrapping here rather than at each call site keeps that from
    reappearing every time a caption is edited.

    The returned top is measured, not guessed: line height comes from the
    point size and the FIGURE'S OWN height, so a caption that grows by two
    lines pushes the axes down instead of landing on their titles. A constant
    was wrong the moment two figures had different heights.
    """
    import textwrap

    lines = []
    for para in body.split("\n"):
        lines += textwrap.wrap(para, chars) or [""]
    fig.suptitle(title, x=.012, y=y_title, ha="left", va="top",
                 fontsize=16.5, fontweight="semibold")
    y_body = y_title - (16.5 * 1.6) / (fig.get_figheight() * 72)
    fig.text(.012, y_body, "\n".join(lines), ha="left", va="top",
             fontsize=fontsize, color=INK2, linespacing=linespacing)
    line_h = (fontsize * linespacing) / (fig.get_figheight() * 72)
    return max(.50, y_body - line_h * len(lines) - .014)
