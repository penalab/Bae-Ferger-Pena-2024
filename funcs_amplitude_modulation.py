from typing import TypedDict
import numpy as np
import numpy.typing as npt

from powltools.analysis.spike_aggregators import SpikeTimesType


class PhaseLockingDataAM(TypedDict):
    spike_angles: npt.NDArray[np.float64]
    vector_strength: float
    mean_phase: float
    p: float


def get_phaselocking_am_stimuli(
    spiketrains: SpikeTimesType,
    modulation_frequency: float,
) -> PhaseLockingDataAM:
    period = 1 / modulation_frequency
    spike_angles = np.concatenate(
        [
            (spiketrain % period) * modulation_frequency * 2 * np.pi
            for spiketrain in spiketrains
        ]
    )
    # Move values in between -pi to +pi
    spike_angles = spike_angles - np.pi

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
    pval = np.exp(
        np.sqrt(1 + 4 * total_n + 4 * (total_n**2 - rayleigh_r**2)) - (1 + 2 * total_n)
    )
    return {
        "spike_angles": spike_angles,
        "vector_strength": vector_strength,
        "mean_phase": mean_angle,
        "p": pval,
    }
