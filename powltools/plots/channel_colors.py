from typing import Callable
import numpy as np
from matplotlib import colormaps
from matplotlib.typing import ColorType
from ..analysis.electrode_mapping import NeuroNexus16Channel as NN16


class ChannelColors:

    def __init__(
        self,
        n_channels: int = 16,
        key: Callable[[int], int] = NN16.sort_deepest_first,
        n_crop_low: int = 0,
        n_crop_high: int = 0,
        cmap="gist_rainbow_r",
    ) -> None:
        self.n_channels = n_channels
        self.key = key
        self.cmap = colormaps.get_cmap(cmap)
        n_crop = n_crop_low + n_crop_high
        crop_slice = slice(
            n_crop_low if n_crop_low else None, -n_crop_high if n_crop_high else None
        )
        self.colors: list[ColorType] = [
            self.cmap(val) for val in np.linspace(0, 1, n_channels + n_crop)[crop_slice]
        ]

    def __getitem__(self, channel_number: int) -> ColorType:
        return self.colors[self.key(channel_number) % self.n_channels]


def channel_colors_example(**channel_colors_kwargs):
    from .nice_figure import NiceFigure

    channel_numbers = list(range(1, 17))
    channel_numbers.sort(key=NN16.sort_deepest_first)

    nfig = NiceFigure(
        1,
        len(channel_numbers),
        individual_height_inch=0.3,
        individual_width_inch=0.3,
        left_inch=0.1,
        right_inch=0.1,
        bottom_inch=0.1,
        wspace_inch=0.0,
        hspace_inch=0.3,
        top_inch=0.6,
    )
    colors = ChannelColors(**channel_colors_kwargs)
    for channel_number, ax in zip(channel_numbers, nfig.iteraxs()):
        ax.set_facecolor(colors[channel_number])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{channel_number}", loc="center", pad=2.0, fontweight="bold")

    nfig.title("Channel Colors", y_inch=0.1)
    nfig.axg.set_title("Tip", loc="left", pad=20, fontsize=10, fontstyle="italic")
    nfig.axg.set_title("Base", loc="right", pad=20, fontsize=10, fontstyle="italic")
    return nfig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    channel_colors_example()
    plt.show()
