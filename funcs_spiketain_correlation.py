from typing import TypedDict
import numpy as np
import numpy.typing as npt
import scipy.signal


def cross_correlation(
    spiketrains1: npt.NDArray[np.int_],
    spiketrains2: npt.NDArray[np.int_],
    lags: npt.NDArray[np.float_],
):
    xcorr = np.asarray(
        [
            (
                scipy.signal.correlate(psth_ot, psth_fix)
                if np.any(psth_ot)
                else np.zeros(lags.size)
            )
            for psth_ot, psth_fix in zip(spiketrains1, spiketrains2)
        ]
    )

    shuffled_ind = np.arange(1, len(spiketrains1) - 1)
    xcorr_shuff = np.asarray(
        [
            (
                scipy.signal.correlate(psth_ot, psth_fix)
                if np.any(psth_ot)
                else np.zeros(lags.size)
            )
            for psth_ot, psth_fix in zip(spiketrains1[shuffled_ind], spiketrains2[:-1])
        ]
    )

    mean_xcorr = smooth(np.nanmean(xcorr, axis=0), 5)
    mean_shuff = smooth(np.nanmean(xcorr_shuff, axis=0), 5)

    return mean_xcorr, mean_shuff


def smooth(
    signal: npt.NDArray[np.float_], window_size: int = 5
) -> npt.NDArray[np.float_]:
    """Smooth a signal with a moving rectangular window.

    Implements equivalent functionality MATLAB's smooth(a, window_size, 'moving')

    On both ends equivalently, average with a symmetric window as large as possible:

    out[0] = a[0]
    out[1] = sum(a[0:3]) / 3
    out[2] = sum(a[0:5]) / 5
    ...

    Parameters
    ----------
    a: NDArray
        1-D array containing the data to be smoothed
    window_size : int
        Smoothing window size needs, which must be odd number,
        as in the original MATLAB implementation

    Returns
    -------
    NDArray
        The smoothed signal

    Notes
    -----
    Implementation adapted from: https://stackoverflow.com/a/40443565
    """
    s = np.convolve(signal, np.ones(window_size, dtype="float"), "valid") / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(signal[: window_size - 1])[::2] / r
    stop = (np.cumsum(signal[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, s, stop))


class CCGpeakData(TypedDict):
    peak_corr: float
    peak_lag: float
    baseline_mean: float
    baseline_std: float
    peak_area: float
    peak_width: float


def get_peak(
    ccg: npt.NDArray[np.float_],
    lags: npt.NDArray[np.float_],
    lag_window: float = 0.015,
    baseline_window: float = 0.050,
) -> CCGpeakData:
    """Get the relevant peak values of a cross-correlogram (CCG)

    Search for the peak value is restricted to lags within [-lag_window, +lag_window], inclusive.

    Parameters
    ----------
    ccg : array
        The cross-correlogram, like returned from scipy.signal.correlate
    lags : array
        The corresponding lags, like returned from scipy.signal.correlation_lags
    lags_window : float
        Defines the range of lags (around 0) where the peak is allowed, same unit as lags
    baseline_window : float
        Defines the range of lags (from both end) to retrieve the CCG baseline

    Returns
    -------
    dict
        peak_corr
            The peak correlation value, from which the baseline mean was already subtracted
        peak_lag
            The lag at which the peak was found
        baseline_mean
            Mean of the baseline
        baseline_std
            Standard deviation of the baseline
        peak_width
            Half-height width of the
        peak_area
            Area under peak, within half-height width
    """
    window_start = np.searchsorted(lags, -lag_window)
    window_stop = np.searchsorted(lags, +lag_window) + 1
    # Number of elements for baseline on each end:
    baseline_n = int(np.unique(np.round(baseline_window / np.diff(lags))))
    baseline = np.concatenate((ccg[:baseline_n], ccg[-baseline_n:]))
    peak_ind = window_start + np.nanargmax(ccg[window_start:window_stop])
    baseline_mean: float = np.mean(baseline).item()

    # Half width and area under the curve
    halfmax = (ccg[peak_ind] + baseline_mean) / 2
    left_ind = peak_ind - np.argmax(ccg[peak_ind::-1] <= halfmax)
    right_ind = peak_ind + np.argmax(ccg[peak_ind::+1] <= halfmax)
    area_under_peak = np.trapz(ccg[left_ind:right_ind], x=lags[left_ind:right_ind])
    peak_width = lags[right_ind] - lags[left_ind]

    return {
        "peak_corr": ccg[peak_ind] - baseline_mean,
        "peak_lag": lags[peak_ind],
        "baseline_mean": baseline_mean,
        "baseline_std": np.std(baseline).item(),
        "peak_area": area_under_peak,
        "peak_width": peak_width,
    }
