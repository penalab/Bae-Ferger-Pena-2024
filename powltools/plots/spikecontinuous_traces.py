from typing import Optional, cast
from itertools import batched
import numpy as np
import numpy.typing as npt
import matplotlib.transforms
from matplotlib.axes import Axes

from ..analysis.electrode_mapping import channel2region
from ..analysis.adhoc_recording import AdhocRecording
from ..analysis.recording import SpikeTimesType
from ..analysis.electrode_mapping import NeuroNexus16Channel as NN16

from .nice_figure import NiceFigure


def figure_spikecontinuous_traces_batched(
    rec: AdhocRecording,
    max_time: float = 10.0,
) -> list[NiceFigure]:
    all_channels = rec.channel_numbers()
    nice_figures: list[NiceFigure] = []
    for channels in batched(all_channels, n=16):
        nice_figures.append(
            figure_spikecontinuous_traces(
                rec, channels=list(channels), max_time=max_time
            )
        )
        nice_figures[-1].meta["channels"] = list(channels)
    return nice_figures


def figure_spikecontinuous_traces(
    rec: AdhocRecording,
    channels: Optional[list[int]] = None,
    max_time=10.0,
):
    if channels is None:
        channels = rec.channel_numbers()

    channels.sort(key=NN16.sort_deepest_first, reverse=True)

    region = " & ".join(set(channel2region(rec, c) for c in channels))

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
    nfig.xlabel("Recording time [seconds]")
    nfig.ylabel("Voltage [µV]")
    nfig.title(f"SRF {rec.session_id} ({channels[0]}-{channels[-1]} {region})")

    nfig.axs[0, 0].set_xlim(0, max_time)
    nfig.axs[0, 0].set_ylim(-0.00025, +0.00025)
    for ax in nfig.axs.flatten():
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x*1e6:+.0f}")

    samplingrate = rec.get_samplingrate("traces")
    plot_t = np.arange(int(samplingrate * max_time)) / samplingrate

    for channel_number, ax in zip(channels, nfig.axs.flatten()):

        ax = cast(Axes, ax)
        ax.plot(
            plot_t,
            rec.spike_continuous(channel_number).continuous_signal[: plot_t.size],
            color=".3",
        )
        ax.axhline(rec.get_threshold(channel_number) * 1e-6, color="r")
        ax.eventplot(
            _spikes_in_range(rec, channel_number, plot_t[-1]),
            color="k",
            lineoffsets=0.05,
            linelengths=0.080,
            transform=matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes
            ),
        )

        ax.set_title(f"{channel_number}", loc="left", pad=2, fontweight="bold")

        ax.text(
            1.0,
            1.01,
            f"{rec.get_threshold(channel_number):.1f}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    return nfig


def _spikes_in_range(
    rec: AdhocRecording, channel_number: int, max_time: float
) -> npt.NDArray[np.float_]:
    spikes_lst: list[float] = []
    for trial_index in sorted(rec.spike_trains(channel_number).keys()):
        st = rec.spike_trains(channel_number)[trial_index]
        start, stop = rec.spike_continuous(channel_number).trials_start_stop[
            trial_index
        ]
        spikes_lst.extend(st + start / rec.samplingrate)
        if stop > max_time * rec.samplingrate:
            break
    spikes: SpikeTimesType = np.asarray(spikes_lst)
    return spikes[spikes <= max_time]
