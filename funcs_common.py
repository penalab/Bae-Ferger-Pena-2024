# Builtin
import glob
from functools import cache
import pathlib
from typing import Callable, Iterator, TypedDict, cast

# 3rd party
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

# Custom
from powltools.analysis.recording import Recording
from powltools.analysis.spike_aggregators import SpikeTimesType
from powltools.analysis.stim_aggregators import StimParamFunc, StimulusParams
from powltools.analysis.stim_aggregators import stim_delay, stim_len, stim_level


@cache
def make_psth_bins(
    stimdelay: float,
    stimduration: float,
    binsize: float = 0.001,
    offset: float = 0.050,
) -> npt.NDArray[np.float64]:
    bins = np.arange(
        stimdelay + offset,
        stimdelay + stimduration + binsize,
        binsize,
        dtype=np.float64,
    )
    bins.setflags(write=False)
    return bins


@cache
def make_psth_bins_base(
    stimdelay: float,
    stimduration: float,
    binsize: float = 0.001,
    offset: float = 0.050,
) -> npt.NDArray[np.float64]:
    bins = np.arange(
        stimdelay - stimduration,
        stimdelay - offset + binsize,
        binsize,
        dtype=np.float64,
    )
    bins.setflags(write=False)
    return bins


def stim_psth_bins(stim_params: StimulusParams) -> npt.NDArray[np.float_]:
    return make_psth_bins(
        stim_delay(stim_params), stim_len(stim_params), binsize=0.001, offset=0.050
    )


def base_psth_bins(stim_params: StimulusParams) -> npt.NDArray[np.float_]:
    return make_psth_bins_base(
        stim_delay(stim_params), stim_len(stim_params), binsize=0.001, offset=0.050
    )


def binary_spiketrain(
    spikes: SpikeTimesType, bins: npt.NDArray[np.float_]
) -> npt.NDArray[np.int_]:
    return np.histogram(spikes, bins)[0].astype(bool).astype(int)


class SessionValues(TypedDict):
    date: str
    owl: int
    dirname: str
    channels: list[int]
    filenames: list[str]


def iter_sessions(
    df: pd.DataFrame,
    filename_filter: Callable[[str], bool],
    data_dir: pathlib.Path,
) -> Iterator[SessionValues]:
    """Session-wise iterator with filtered file list

    Parameters
    ----------
    df : pandas.DataFrame
        `date` (e.g. "2023-05-15") and `owl` (e.g. 33) are index columns
        and denote the session date and owl ring number, respectively.
        `channel` (e.g. 5) indicates the channel number of auditory units.
    filename_filter : Callable[[str], bool]
        A function that takes one HDF5 filename (.h5) and returns a boolean
        whether to include this file in the list.
    data_dir : pathlib.Path
        The path to the data directory. Inside this, sessions are saved in
        subdirectories following the `[YYYYMMDD]_[OWL]_awake` naming scheme,
        where [YYYYMMDD] is the date and [OWL] is the owl ring number.

    Yields
    ------
    A dictionary with session information, including the date, owl number,
    session directory (relative to data_dir), a list of channel numbers and
    the filtered list of filenames.
    """
    for (index_date, index_owl), channelseries in df.groupby(["date", "owl"]):
        # print(index)
        # = ('2023-05-15', 33)
        index_dirname = f"{index_date.replace('-', '')}_{index_owl}_awake"
        channels: list[int] = sorted(np.asarray(channelseries).flatten().tolist())
        # print(channels)
        filenames = list(
            filter(
                filename_filter,
                glob.glob(str(data_dir / index_dirname / "*.h5")),
            )
        )
        yield {
            "date": index_date,
            "owl": index_owl,
            "dirname": index_dirname,
            "channels": channels,
            "filenames": filenames,
        }


class FixedVarStimuli(TypedDict):
    fixed_index: int
    fixed_value: str | int | float
    varying_index: int
    varying_values: list[str | int | float]


def fixed_var_stimuli(
    rec: Recording, func: StimParamFunc = stim_level
) -> FixedVarStimuli:
    with rec.powlfile as pf:
        stimulus_indexes = {
            int(k) for k in cast(h5py.Group, pf["trials/0/stimuli"]).keys()
        }
        if not len(stimulus_indexes) == 2:
            raise ValueError(
                "Only one stimulus found, expected two."
                if len(stimulus_indexes) == 1
                else "More than two stimuli found, expected two."
            )
        unique_values = {
            stimulus_index: set(
                rec.aggregate_stim_params(func, stimulus_index=stimulus_index)
            )
            for stimulus_index in stimulus_indexes
        }
    # Find the first stimulus_index with a single unique value
    fixed_index = next(
        stimulus_index
        for stimulus_index, uvalues in unique_values.items()
        if len(uvalues) == 1
    )
    varying_index = (stimulus_indexes - {fixed_index}).pop()
    return {
        "fixed_index": fixed_index,
        "fixed_value": unique_values[fixed_index].pop(),
        "varying_index": varying_index,
        "varying_values": sorted(unique_values[varying_index]),
    }
