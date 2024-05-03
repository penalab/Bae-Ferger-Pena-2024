from itertools import batched
from typing import Literal, Optional, Any, cast
from matplotlib.figure import Figure
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..analysis.recording import Recording
from ..analysis.stim_aggregators import stim_position, stim_delay
from ..analysis.spike_aggregators import spike_count_spontaneous
from ..analysis.grouping import group_by_multiparam
from ..analysis.adhoc_recording import AdhocRecording
from ..analysis.electrode_mapping import NeuroNexus16Channel as NN16

from .nice_figure import NiceFigure


def figure_spatial_receptive_field_batched(rec: Recording) -> list[NiceFigure]:
    all_channels = rec.channel_numbers()
    nice_figures: list[NiceFigure] = []
    for channels in batched(all_channels, n=16):
        nice_figures.append(
            figure_spatial_receptive_field(rec, channels=list(channels))
        )
        nice_figures[-1].meta["channels"] = list(channels)
    return nice_figures


def figure_spatial_receptive_field(
    rec: Recording,
    channels: Optional[list[int]] = None,
) -> NiceFigure:

    if channels is None:
        channels = rec.channel_numbers()

    channels.sort(key=NN16.sort_deepest_first, reverse=True)

    nfig = NiceFigure(
        n_rows=4,
        n_cols=4,
        individual_width_inch=2.0,
        individual_height_inch=1.4,
        left_inch=1.0,
        right_inch=0.3,
        bottom_inch=0.6,
        top_inch=0.8,
        sharex=True,
        sharey=True,
    )
    nfig.xlabel("Azimuth [deg]")
    nfig.ylabel("Elevation [deg]")
    nfig.title(f"SRF {rec.session_id}")

    positions = rec.aggregate_stim_params(stim_position)

    ax: Axes
    for channel_number, ax in zip(channels, nfig.axs.flatten()):
        spont_rate = np.mean(
            rec.aggregrate_spikes(
                spike_count_spontaneous,
                rec.aggregate_stim_params(stim_delay),
                channel_number=channel_number,
            )
        ).item()

        trial_responses = rec.response_rates(
            channel_number=channel_number, stimulus_index=0
        )
        azimuths, elevations, response_rates = np.asarray(
            [
                [azi, ele, resp.mean()]
                for (azi, ele), resp in group_by_multiparam(
                    trial_responses, positions
                ).items()
            ]
        ).T

        single_spatial_receptive_field(
            azimuths,
            elevations,
            response_rates,
            ax=ax,
            spont_rate=spont_rate,
        )
        ax.set_title(f"{channel_number}", loc="left", pad=2.0, fontweight="bold")

        if isinstance(rec, AdhocRecording):
            ax.text(
                1.0,
                1.01,
                f"{rec.get_threshold(channel_number):.1f}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )

    return nfig


def single_spatial_receptive_field(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    responses: np.ndarray,
    *,
    ax: Axes | None = None,
    spont_rate: float | None = None,
    marker_on_max: bool = True,
    marker_above: float | Literal[False] = 0.9,
):
    """Plotting spatial receptive field"""
    if ax is None:
        fig, ax = cast(tuple[Figure, Axes], plt.subplots(1, 1))
    else:
        fig = cast(Figure, ax.figure)
    finite_responses = np.isfinite(responses)
    if np.any(~finite_responses):
        missing_azimuths = azimuths[~finite_responses]
        missing_elevations = elevations[~finite_responses]
        azimuths = azimuths[finite_responses]
        elevations = elevations[finite_responses]
        responses = responses[finite_responses]
    order = np.argsort(responses)
    azimuths = np.asarray(azimuths)[order]
    elevations = np.asarray(elevations)[order]
    responses = np.asarray(responses)[order]
    srf = ax.scatter(
        x=azimuths,
        y=elevations,
        c=responses,
        cmap=mpl.colormaps["Greys"],
        vmin=0.0,
        vmax=100 if responses[-1] > 100 else None,
    )
    cb = fig.colorbar(srf)
    if spont_rate is not None:
        cb.ax.plot(
            cb.ax.get_xlim()[-1],
            spont_rate,
            ls="none",
            marker="<",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=4,
            clip_on=False,
            zorder=10,
        )
    if marker_on_max:
        ax.scatter(
            x=azimuths[-1],
            y=elevations[-1],
            c="w",
            marker="+",
        )
    if marker_above is not None:
        mask_above = np.searchsorted(responses, marker_above * responses[-1])
        ax.scatter(
            x=azimuths[mask_above:-1],
            y=elevations[mask_above:-1],
            c="w",
            marker=".",
        )
    # Plot missing:
    if np.any(~finite_responses):
        # ax.scatter(x=azimuths[missing], y=elevations[missing], c="w", edgecolors=".6")
        ax.scatter(
            x=missing_azimuths,
            y=missing_elevations,
            s=9,
            c="k",
            marker="x",
            linewidths=0.5,
        )
