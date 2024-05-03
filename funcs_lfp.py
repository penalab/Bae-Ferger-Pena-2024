from typing import TypedDict
import numpy as np
import numpy.typing as npt
import scipy.signal

from powltools.analysis.lfp_aggregators import LFPType
from powltools.analysis.spike_aggregators import SpikeTimesType


def get_power(
    trial_lfp: npt.NDArray[np.float64],
    baseline_start: float,
    baseline_stop: float,
    stim_start: float,
    stim_stop: float,
    samplingrate: float = 1000.0,
):
    s_hilb = np.abs(np.asarray(scipy.signal.hilbert(trial_lfp)))
    hilb_power = s_hilb**2

    lfp_times = np.arange(trial_lfp.size) / samplingrate

    baseline_slice = slice(
        np.searchsorted(lfp_times, baseline_start),
        np.searchsorted(lfp_times, baseline_stop),
    )
    stim_slice = slice(
        np.searchsorted(lfp_times, stim_start), np.searchsorted(lfp_times, stim_stop)
    )
    baseline_pow = hilb_power[baseline_slice]
    stim_pow = hilb_power[stim_slice]

    baseline_rms = np.sqrt(np.sum(baseline_pow**2) / baseline_pow.size)
    stim_rms = np.sqrt(np.sum(stim_pow**2) / stim_pow.size)

    rel_db = 20 * np.log(stim_rms / baseline_rms)
    return rel_db


class PhaseLockingData(TypedDict):
    vector_strength: float
    mean_phase: float
    p: float


def get_phaselocking(
    filt_lfp: LFPType,
    spiketrains: SpikeTimesType,
    lfp_samplingrate: float = 1000.0,
) -> PhaseLockingData:
    assert filt_lfp.ndim == 2
    lfp_times = np.arange(filt_lfp.shape[1]) / lfp_samplingrate

    s_hilb = np.asarray(scipy.signal.hilbert(filt_lfp))
    s_angles = np.angle(s_hilb)

    spike_angles = np.concatenate(
        [
            np.interp(trial_spiketimes, lfp_times, trial_phases)
            for trial_spiketimes, trial_phases in zip(spiketrains, s_angles)
        ]
    )
    spike_phases: npt.NDArray[np.complex128] = (
        np.cos(spike_angles) + np.sin(spike_angles) * 1j
    )

    mean_vector = np.mean(spike_phases).item()
    mean_angle = np.angle(mean_vector).item()
    vector_strength: float = np.abs(mean_vector)

    ## Code for rayleigh significance
    total_n = spike_phases.size
    # compute Rayleigh's R (equ. 27.1)
    rayleigh_r = total_n * vector_strength
    # compute p value using approxation in Zar, p. 617
    pval: float = np.exp(
        np.sqrt(1 + 4 * total_n + 4 * (total_n**2 - rayleigh_r**2)) - (1 + 2 * total_n)
    )
    return {"vector_strength": vector_strength, "mean_phase": mean_angle, "p": pval}
