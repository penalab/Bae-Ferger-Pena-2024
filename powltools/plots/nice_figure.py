from functools import cached_property
from typing import Any, Iterator
from matplotlib.axes import Axes
from matplotlib.text import Text
import matplotlib.pyplot as plt


class NiceFigure:
    def __init__(
        self,
        n_rows=1,
        n_cols=1,
        *,
        individual_width_inch=3.0,
        wspace_inch=0.5,
        left_inch=2.0,
        right_inch=0.5,
        individual_height_inch=3.0,
        hspace_inch=0.5,
        bottom_inch=1.0,
        top_inch=0.5,
        sharex=False,
        sharey=False,
        meta=None,
    ):
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._individual_width_inch = individual_width_inch
        self._wspace_inch = wspace_inch
        self._left_inch = left_inch
        self._right_inch = right_inch
        self._individual_height_inch = individual_height_inch
        self._hspace_inch = hspace_inch
        self._bottom_inch = bottom_inch
        self._top_inch = top_inch
        self._sharex = sharex
        self._sharey = sharey
        self.meta = meta or {}

        self._title: Text | None = None
        self._xlabel: Text | None = None
        self._ylabel: Text | None = None

        self.figsize, self.gridspec_kw = make_dimensions(
            self._n_rows,
            self._n_cols,
            individual_width_inch=self._individual_width_inch,
            wspace_inch=self._wspace_inch,
            left_inch=self._left_inch,
            right_inch=self._right_inch,
            individual_height_inch=self._individual_height_inch,
            hspace_inch=self._hspace_inch,
            bottom_inch=self._bottom_inch,
            top_inch=self._top_inch,
        )

        self.fig, self.axs = plt.subplots(
            self._n_rows,
            self._n_cols,
            figsize=self.figsize,
            gridspec_kw=self.gridspec_kw,
            sharex=self._sharex,
            sharey=self._sharey,
            squeeze=False,
        )

    @property
    def gridspec(self):
        return self.axs[0, 0].get_gridspec()

    @cached_property
    def axg(self) -> Axes:
        # Create Axes for global axis labels
        if self.axs.size == 1:
            return self.axs[0, 0]
        axg = self.fig.add_subplot(
            self.gridspec[:], zorder=20, frame_on=False, xticks=[], yticks=[]
        )
        return axg

    def iteraxs(self) -> Iterator[Axes]:
        for ax in self.axs.flatten():
            yield ax

    def title(self, t: str, y_inch: float = 0.3, **kwargs):
        if self._title is not None:
            self._title.remove()
        h = self.fig.get_figheight()
        kwargs.pop("y", None)
        kwargs = {"fontweight": "bold", "fontsize": 14} | kwargs
        self._title = self.fig.suptitle(t, y=(h - y_inch) / h, **kwargs)
        return self._title

    def xlabel(self, xlabel: str, labelpad: float = 20.0, **kwargs):
        if self._xlabel is not None:
            self._xlabel.remove()
        setter_kwargs: dict[str, Any] = {"fontweight": "bold"} | kwargs
        self._xlabel = self.axg.set_xlabel(xlabel, labelpad=labelpad, **setter_kwargs)
        return self._xlabel

    def ylabel(self, ylabel: str, labelpad: float = 40.0, **kwargs):
        if self._ylabel is not None:
            self._ylabel.remove()
        setter_kwargs: dict[str, Any] = {"fontweight": "bold"} | kwargs
        self._ylabel = self.axg.set_ylabel(ylabel, labelpad=labelpad, **setter_kwargs)
        return self._ylabel


def make_dimensions(
    n_rows=1,
    n_cols=1,
    individual_width_inch=3.0,
    wspace_inch=0.5,
    left_inch=2.0,
    right_inch=0.5,
    individual_height_inch=3.0,
    hspace_inch=0.5,
    bottom_inch=1.0,
    top_inch=0.5,
):
    """Make dimensions (figsize and gridspec_kw) for a subplotted figure from absolute values"""
    wspace = wspace_inch / individual_width_inch
    plot_area_width_inch = individual_width_inch * n_cols + (n_cols - 1) * wspace_inch
    total_width_inch = plot_area_width_inch + left_inch + right_inch
    left = left_inch / total_width_inch
    right = 1 - right_inch / total_width_inch

    hspace = hspace_inch / individual_height_inch
    plot_area_height_inch = individual_height_inch * n_rows + (n_rows - 1) * hspace_inch
    total_height_inch = plot_area_height_inch + bottom_inch + top_inch
    bottom = bottom_inch / total_height_inch
    top = 1 - top_inch / total_height_inch

    return (total_width_inch, total_height_inch), dict(
        left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace
    )
