from typing import Any, Callable
import numpy as np
import numpy.typing as npt

SpikeTimesType = npt.NDArray[np.float_]
SpikeTrainsType = dict[int, SpikeTimesType]
SpikeTimesFunc = Callable[..., Any]


def spike_count_response(st: SpikeTimesType, delay: float, dur: float) -> int:
    return np.searchsorted(st, delay + dur).item() - np.searchsorted(st, delay).item()


def spike_count_spontaneous(
    spiketimes: np.ndarray[Any, np.dtype[np.float_]],
    delay: float,
) -> int:
    return np.searchsorted(spiketimes, delay).item()
