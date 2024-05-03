"""Plotting functions to make my life easier."""

from __future__ import annotations as _annotations

# Builtin
from typing import TypeAlias, Union, cast

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory, ScaledTranslation
from matplotlib.patches import BoxStyle
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Emulate ColorType from matplotlib.typing in later versions
# see: https://matplotlib.org/3.8.0/api/typing_api.html#matplotlib.typing.ColorType
ColorType: TypeAlias = Union[
    tuple[float, float, float],
    str,
    tuple[float, float, float, float],
    tuple[Union[tuple[float, float, float], str], float],
    tuple[tuple[float, float, float, float], float],
]


def plot_bracket(
    ax: Axes,
    left: float,
    right: float,
    text: str = "",
    y: float = 0.8,
    shrink: float = 0.8,
):
    m = (left + right) / 2
    d = right - left
    left = m - (d / 2) * shrink
    right = m + (d / 2) * shrink
    ax.plot(
        [left, left, right, right],
        [y - 0.05, y, y, y - 0.05],
        ls="-",
        lw=1.0,
        color="k",
        transform=(
            blended_transform_factory(ax.transData, ax.transAxes)
            + ScaledTranslation(0, 0, ax.figure.dpi_scale_trans)  # type: ignore
        ),
    )
    if text:
        ax.text(
            m,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            transform=(
                blended_transform_factory(ax.transData, ax.transAxes)
                + ScaledTranslation(0, 2 / 72, ax.figure.dpi_scale_trans)  # type: ignore
            ),
        )


def figure_outline(fig: Figure | None = None):
    if fig is None:
        fig = cast(Figure, plt.gcf())
    with plt.xkcd(0.5):  # type: ignore
        ax0 = fig.add_axes(
            (0.001, 0.001, 0.998, 0.998),
            frameon=True,
            xticks=[],
            yticks=[],
            facecolor="none",
            zorder=-100,
        )
        plt.setp(ax0.spines.values(), color="r", linestyle=":")  # type: ignore


def figure_add_axes_inch(
    fig: Figure,
    left=None,
    width=None,
    right=None,
    bottom=None,
    height=None,
    top=None,
    label=None,
    **kwargs,
) -> Axes:
    """Add Axes to a figure with inch coordinates."""
    # Check number of arguments:
    n_horz_args = sum([left is not None, width is not None, right is not None])
    if not n_horz_args == 2:
        raise ValueError(
            f"Need exactly 2 horizontal arguments, but {n_horz_args} given."
        )
    n_vert_args = sum([bottom is not None, height is not None, top is not None])
    if not n_vert_args == 2:
        raise ValueError(
            f"Need exactly 2 horizontal arguments, but {n_vert_args} given."
        )
    # Unique label:
    if label is None:
        label = f"ax{len(fig.get_axes()):02}"

    # Figure dimensions:
    fig_w, fig_h = fig.get_size_inches()

    # Horizontal:
    if right is None:
        l = left / fig_w
        w = width / fig_w
    elif width is None:
        l = left / fig_w
        w = (fig_w - left - right) / fig_w
    else:  # left is None
        w = width / fig_w
        l = (fig_w - right - width) / fig_w

    # Vertical:
    if top is None:
        b = bottom / fig_h
        h = height / fig_h
    elif height is None:
        b = bottom / fig_h
        h = (fig_h - bottom - top) / fig_h
    else:  # bottom is None
        h = height / fig_h
        b = (fig_h - top - height) / fig_h

    return fig.add_axes((l, b, w, h), label=label, **kwargs)


def figure_add_axes_group_inch(
    fig: Figure,
    nrows=1,
    ncols=1,
    group_top=0.2,
    group_left=0.8,
    individual_width=1.2,
    individual_height=0.8,
    wspace=0.1,
    hspace=0.1,
):
    axs = []
    for kr in range(nrows):
        axs.append([])
        for kc in range(ncols):
            ax = figure_add_axes_inch(
                fig,
                top=group_top + kr * (individual_height + hspace),
                height=individual_height,
                left=group_left + kc * (individual_width + wspace),
                width=individual_width,
            )
            axs[-1].append(ax)
    axs = np.asarray(axs)
    axg = figure_add_axes_inch(
        fig,
        top=group_top,
        height=nrows * individual_height + (nrows - 1) * hspace,
        left=group_left,
        width=ncols * individual_width + (ncols - 1) * wspace,
    )
    plt.setp(axg, frame_on=False, xticks=[], yticks=[], zorder=20)
    return axs, axg


def subplot_indicator(ax: Axes, label=None, fontsize=16, pad_inch=None, **kwargs):
    shift_left = (fontsize / 2 / 72.0) if pad_inch is None else pad_inch
    trans = ax.transAxes + ScaledTranslation(
        -shift_left, +0 / 72.0, ax.figure.dpi_scale_trans  # type: ignore
    )
    if label is None:
        label = ax.get_label()
    if "ha" in kwargs:
        kwargs["horizontalalignment"] = kwargs.pop("ha")
    if "va" in kwargs:
        kwargs["verticalalignment"] = kwargs.pop("va")
    textkwargs = dict(
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )
    textkwargs.update(kwargs)
    ax.text(0.0, 1.0, label, transform=trans, **textkwargs)  # type: ignore


def condition_batch(
    fig: Figure,
    left: float = 0.8,
    top: float = 0.3,
    text: str = "label",
    *,
    fontsize: float | int = 8,
    color="black",
    ha="right",
    va="bottom",
    x_pt=0.0,
    y_pt=0.0,
    pad_pt=4.0,
):
    x_pt = x_pt + pad_pt
    y_pt = y_pt + pad_pt
    fig.text(
        0,
        1,
        text,
        color="white",
        fontsize=fontsize,
        fontweight="bold",
        ha=ha,
        va=va,
        bbox={
            "facecolor": color,
            "alpha": 1,
            "linewidth": 0,
            "boxstyle": BoxStyle(
                "Round",
                pad=pad_pt / fontsize,
                rounding_size=2 * pad_pt / fontsize,
            ),
        },
        transform=fig.transFigure  # type: ignore
        + ScaledTranslation(
            +left + (-1 if ha == "right" else +1) * x_pt / 72,
            -top + (-1 if va == "top" else +1) * y_pt / 72,
            fig.dpi_scale_trans,  # type: ignore
        ),
    )
