# Builtin
from itertools import combinations, repeat
import pathlib

# 3rd party
import numpy as np
import numpy.typing as npt
import scipy.stats
import scipy.signal
import pandas as pd

# Custom
from powltools.io.file import POwlFile
from powltools.analysis.recording import Recording
from powltools.analysis.stim_aggregators import stim_delay, stim_len, stim_level
from powltools.analysis.grouping import group_by_param
from powltools.filters.offlinefilters import bandpass_filter
from funcs_common import iter_sessions
from funcs_common import binary_spiketrain
from funcs_common import make_psth_bins
from funcs_common import stim_psth_bins
from funcs_common import base_psth_bins
from funcs_common import fixed_var_stimuli
from funcs_lfp import get_power
from funcs_lfp import get_phaselocking
from funcs_spiketain_correlation import cross_correlation
from funcs_spiketain_correlation import get_peak
from settings_rawdata_path import datadir_ot, units_ot
from settings_rawdata_path import datadir_icx, units_icx


def main():
    # Single stimulus data, flat noise, curated/clean
    # Contains one line per [date, owl, channel, level]
    OUTDIR = pathlib.Path("./intermediate_results").absolute()
    OUTDIR.mkdir(exist_ok=True)
    for region in ["ot", "icx"]:
        if region == "ot":
            data_dir = datadir_ot
            df = pd.read_csv(units_ot)
            df.set_index(["date", "owl"], inplace=True)
            df.sort_index(inplace=True)
        elif region == "icx":
            data_dir = datadir_icx
            df = pd.read_csv(units_icx)
            df.set_index(["date", "owl"], inplace=True)
            df.sort_index(inplace=True)

        # Single stim:
        print(region, "singlestim_ccg".upper())
        single_ccg_df = singlestim_ccg(df, data_dir)
        single_ccg_df.to_feather(OUTDIR / f"single_ccg_{region}.feather")
        print(region, "singlestim_rlf".upper())
        singlestim_rlf_df = singlestim_rlf(df, data_dir)
        singlestim_rlf_df.to_feather(OUTDIR / f"single_rlf_{region}.feather")
        print(region, "singlestim_gamma_power".upper())
        singlestim_gamma_power_df = singlestim_gamma_power(df, data_dir)
        singlestim_gamma_power_df.to_feather(
            OUTDIR / f"single_gamma_power_{region}.feather"
        )
        # Two Stim:
        print(region, "twostim_ccg".upper())
        twostim_ccg_df = twostim_ccg(df, data_dir)
        twostim_ccg_df.to_feather(OUTDIR / f"twostim_ccg_{region}.feather")
        print(region, "twostim_rlf".upper())
        twostim_rlf_df = twostim_rlf(df, data_dir)
        twostim_rlf_df.to_feather(OUTDIR / f"twostim_rlf_{region}.feather")
        print(region, "twostim_gamma_power".upper())
        twostim_gamma_power_df = twostim_gamma_power(df, data_dir)
        twostim_gamma_power_df.to_feather(
            OUTDIR / f"twostim_gamma_power_{region}.feather"
        )
    return 0


def filter_rate_level_flat(filename: str) -> bool:
    return (
        "rate_level" in filename
        and "amplitude" not in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def filter_relative_level_flat(filename: str) -> bool:
    filename = filename.lower()
    return (
        "relative_intensity" in filename
        and "amplitude" not in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def get_latency(trace, time_bins):
    max_ind = np.argmax(trace)
    half_ind = np.argmax(trace >= trace[max_ind] / 2)
    latency = time_bins[half_ind]
    peak_time = time_bins[max_ind]
    return latency, peak_time


def singlestim_ccg(df, data_dir):
    singlestim_ccg = []

    for session in iter_sessions(
        df, filename_filter=filter_rate_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels: {session['channels']!r})"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore

            psth_bins = rec.aggregate_stim_params(stim_psth_bins)
            base_bins = rec.aggregate_stim_params(base_psth_bins)

            varying_levels = np.array(
                rec.aggregate_stim_params(stim_level, stimulus_index=0)
            )
            unique_levels = np.unique(varying_levels)
            fixed_azimuths = np.array(
                rec.aggregate_stim_params(lambda params: params["azi"])
            )
            fixed_elevations = np.array(
                rec.aggregate_stim_params(lambda params: params["ele"])
            )

            # Binary spiketrains during and before stimuli for all channels
            stim_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan
                    )
                )
                for chan in channels
            }
            base_spiketrains = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, base_bins, channel_number=chan
                    )
                )
                for chan in channels
            }

            for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
                for level in unique_levels:
                    if not level in (-5, -20):
                        # We only need the levels that were used for drivers in competition
                        continue

                    # Boolean array to select trials of this level:
                    mask = varying_levels == level
                    # Data for this condition:
                    psth_u1 = stim_spiketrains[chan1][mask]
                    psth_u2 = stim_spiketrains[chan2][mask]
                    base_u1 = base_spiketrains[chan1][mask]
                    base_u2 = base_spiketrains[chan2][mask]

                    lags = (
                        scipy.signal.correlation_lags(
                            psth_u1.shape[1], psth_u2.shape[1]
                        )
                        * 0.001
                    )

                    # Mean firing rates:
                    resp_rate_u1: float = psth_u1.sum() / psth_u1.shape[0]
                    resp_rate_u2: float = psth_u2.sum() / psth_u2.shape[0]
                    base_rate_u1: float = base_u1.sum() / base_u1.shape[0]
                    base_rate_u2: float = base_u2.sum() / base_u2.shape[0]
                    # Geometric means:
                    gm_resp_rate = (resp_rate_u1 * resp_rate_u2) ** 0.5
                    gm_base_rate = (base_rate_u1 * base_rate_u2) ** 0.5
                    # Exclude of levels with no response
                    if gm_resp_rate <= gm_base_rate:
                        continue

                    ccg, shuff_ccg = cross_correlation(psth_u1, psth_u2, lags)

                    # Normalize by (geometric) mean response rate and psth length
                    norm_ccg = (
                        (ccg - np.mean(ccg)) / (gm_resp_rate) / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])
                    norm_shuff = (
                        (shuff_ccg - np.mean(shuff_ccg))
                        / (gm_resp_rate)
                        / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])

                    smscorrected = norm_ccg - norm_shuff

                    ccg_peak = get_peak(
                        smscorrected, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.050
                    )

                    if ccg_peak["peak_corr"] > 5 * ccg_peak["baseline_std"]:
                        tmp = {
                            "date": session["date"],
                            "owl": session["owl"],
                            "channel1": chan1,
                            "channel2": chan2,
                            "azimuth": fixed_azimuths[0],
                            "elevation": fixed_elevations[0],
                            "intensity": level,
                            "xcorr_peak": ccg_peak["peak_corr"],
                            "peak_time": ccg_peak["peak_lag"],
                            "synchrony_val": ccg_peak["peak_area"],
                            "xcorr_width": ccg_peak["peak_width"],
                            "xcorr_peak_shuff": ccg_peak_shuff["peak_corr"],
                            "peak_time_shuff": ccg_peak_shuff["peak_lag"],
                            "synchrony_val_shuff": ccg_peak_shuff["peak_area"],
                            "xcorr_width_shuff": ccg_peak_shuff["peak_width"],
                            "hemisphere": hemisphere,
                            "stimtype": "single",
                            "ccg": smscorrected,
                            "ccg_shuff": norm_shuff,
                        }
                        singlestim_ccg.append(tmp)

    single_ccg_df = pd.DataFrame(singlestim_ccg)
    return single_ccg_df


def singlestim_rlf(df, data_dir):
    singlestim_rlf = []

    for session in iter_sessions(
        df, filename_filter=filter_rate_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            trial_delays = set(rec.aggregate_stim_params(stim_delay, stimulus_index=0))
            trial_durations = set(rec.aggregate_stim_params(stim_len, stimulus_index=0))
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            if any([trial_delays, trial_durations]):
                raise ValueError(
                    "Stimulus delay or durcation not the same for all trials"
                )
            del trial_delays, trial_durations

            latency_bins = make_psth_bins(0, delay + duration, binsize=0.001, offset=0)

            trial_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain,
                        repeat(latency_bins),
                        channel_number=chan,
                    )
                )
                for chan in list(channels)
            }

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=0,
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=0,
                )
            )
            azimuth = fixed_azimuths.pop()
            elevation = fixed_elevations.pop()
            if any([fixed_azimuths, fixed_elevations]):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations

            for chan in channels:
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                trial_levels = np.array(
                    rec.aggregate_stim_params(stim_level, stimulus_index=0)
                )
                resp_by_level = group_by_param(resp, trial_levels)

                psth_by_level = group_by_param(trial_spiketrains[chan], trial_levels)
                # print(psth)
                # print(psth_by_level)
                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
                    time_firstspike, time_peak = get_latency(
                        mean_psth[latency_bins[:-1] >= delay],
                        latency_bins[latency_bins >= delay],
                    )
                    time_firstspike = time_firstspike - delay
                    time_peak = time_peak - delay

                    # if (
                    #     (time_firstspike < 0.0)
                    #     or (time_peak < 0.0)
                    #     or (time_firstspike > 0.040)
                    #     or (time_peak > 0.040)
                    # ):
                    #     continue
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "intensity": level,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "psth": mean_psth,
                        "first_spike_latency": time_firstspike,
                        "max_peak_latency": time_peak,
                        "hemisphere": hemisphere,
                        "stimtype": "singlestim",
                    }
                    singlestim_rlf.append(tmp)

    singlestim_rlf_df = pd.DataFrame(singlestim_rlf)
    return singlestim_rlf_df


def singlestim_gamma_power(df, data_dir):
    singlestim_gamma_power = []

    for session in iter_sessions(
        df, filename_filter=filter_rate_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore

            trial_levels = rec.aggregate_stim_params(stim_level, stimulus_index=0)
            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=0,
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=0,
                )
            )
            azimuth = fixed_azimuths.pop()
            elevation = fixed_elevations.pop()
            if any([fixed_azimuths, fixed_elevations]):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations

            for chan in channels:
                stim_spikes = np.array(
                    rec.stim_spiketrains(channel_number=chan, ignore_onset=0.050),
                    dtype="object",
                )
                spiketrains_by_level = group_by_param(stim_spikes, trial_levels)

                lfp_arr = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                lfp_arr = lfp_arr - np.mean(lfp_arr, axis=0)
                bandpass_lfp = bandpass_filter(
                    lfp_arr,
                    20,
                    50,
                    fs=lfp_samplingrate,
                    order=1,
                )

                trial_power = np.array(
                    [
                        get_power(
                            bandpass_lfp[trial_index],
                            baseline_start=trial_delays[trial_index]
                            - trial_durations[trial_index],
                            baseline_stop=trial_delays[trial_index] - 0.05,
                            stim_start=trial_delays[trial_index] + 0.05,
                            stim_stop=trial_delays[trial_index]
                            + trial_durations[trial_index],
                            samplingrate=lfp_samplingrate,
                        )
                        for trial_index in rec.powlfile.trial_indexes
                    ]
                )
                power_by_level = group_by_param(trial_power, trial_levels)
                lfp_by_level = group_by_param(bandpass_lfp, trial_levels)

                for level, level_power in power_by_level.items():
                    level_phaselocking = get_phaselocking(
                        lfp_by_level[level], spiketrains_by_level[level]
                    )
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "intensity": level,
                        "gammapower": np.mean(level_power),
                        "gammapower_sem": scipy.stats.sem(level_power),
                        "gamma_plv": level_phaselocking["vector_strength"],
                        "gamma_plv_angle": level_phaselocking["mean_phase"],
                        "gamma_plv_p": level_phaselocking["p"],
                        "hemisphere": hemisphere,
                        "stimtype": "singlestim",
                    }
                    singlestim_gamma_power.append(tmp)
    singlestim_gamma_power_df = pd.DataFrame(singlestim_gamma_power)
    return singlestim_gamma_power_df


def twostim_ccg(df, data_dir):
    twostim_ccg = []

    for session in iter_sessions(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)

            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore

            psth_bins = rec.aggregate_stim_params(stim_psth_bins)
            base_bins = rec.aggregate_stim_params(base_psth_bins)

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels: npt.NDArray[np.float_] = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )
            unique_levels: list[float] = fixed_varying["varying_values"]  # type: ignore
            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            # Binary spiketrains during and before stimuli for all channels
            stim_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan
                    )
                )
                for chan in channels
            }
            base_spiketrains = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, base_bins, channel_number=chan
                    )
                )
                for chan in channels
            }

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
                for level in unique_levels:
                    # Boolean array to select trials of this level:
                    mask = varying_levels == level
                    # Data for this condition:
                    psth_u1 = stim_spiketrains[chan1][mask]
                    psth_u2 = stim_spiketrains[chan2][mask]
                    base_u1 = base_spiketrains[chan1][mask]
                    base_u2 = base_spiketrains[chan2][mask]

                    lags = (
                        scipy.signal.correlation_lags(
                            psth_u1.shape[1], psth_u2.shape[1]
                        )
                        * 0.001
                    )

                    # Mean firing rates:
                    resp_rate_u1: float = psth_u1.sum() / psth_u1.shape[0]
                    resp_rate_u2: float = psth_u2.sum() / psth_u2.shape[0]
                    base_rate_u1: float = base_u1.sum() / base_u1.shape[0]
                    base_rate_u2: float = base_u2.sum() / base_u2.shape[0]
                    # Geometric means:
                    gm_resp_rate = (resp_rate_u1 * resp_rate_u2) ** 0.5
                    gm_base_rate = (base_rate_u1 * base_rate_u2) ** 0.5
                    # Exclude of levels with no response
                    if gm_resp_rate <= gm_base_rate:
                        continue

                    ccg, shuff_ccg = cross_correlation(psth_u1, psth_u2, lags)

                    # Normalize by (geometric) mean response rate and psth length
                    norm_ccg = (
                        (ccg - np.mean(ccg)) / (gm_resp_rate) / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])
                    norm_shuff = (
                        (shuff_ccg - np.mean(shuff_ccg))
                        / (gm_resp_rate)
                        / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])

                    smscorrected = norm_ccg - norm_shuff

                    ccg_peak = get_peak(
                        smscorrected, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.050
                    )

                    relative_level = level - fixed_level

                    if ccg_peak["peak_corr"] > 5 * ccg_peak["baseline_std"]:
                        tmp = {
                            "date": session["date"],
                            "owl": session["owl"],
                            "channel1": chan1,
                            "channel2": chan2,
                            "fixedazi": fixed_azimuth,
                            "fixedele": fixed_elevation,
                            "fixedintensity": fixed_level,
                            "varyingazi": varying_azimuth,
                            "varyingele": varying_elevation,
                            "varyingintensity": level,
                            "relative_level": relative_level,
                            "xcorr_peak": ccg_peak["peak_corr"],
                            "peak_time": ccg_peak["peak_lag"],
                            "synchrony_val": ccg_peak["peak_area"],
                            "xcorr_width": ccg_peak["peak_width"],
                            "xcorr_peak_shuff": ccg_peak_shuff["peak_corr"],
                            "peak_time_shuff": ccg_peak_shuff["peak_lag"],
                            "synchrony_val_shuff": ccg_peak_shuff["peak_area"],
                            "xcorr_width_shuff": ccg_peak_shuff["peak_width"],
                            "hemisphere": hemisphere,
                            "ipsi_contra": ipsi_contra,
                            "stimtype": "twostim",
                            "ccg": smscorrected,
                            "ccg_shuff": norm_shuff,
                        }
                        twostim_ccg.append(tmp)

    twostim_ccg_df = pd.DataFrame(twostim_ccg)
    return twostim_ccg_df


def twostim_rlf(df, data_dir):
    twostim_rlf = []

    for session in iter_sessions(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for chan in channels:
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                resp_by_level = group_by_param(resp, varying_levels)
                for level, level_resp in resp_by_level.items():
                    relative_level = level - fixed_level
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                    }
                    twostim_rlf.append(tmp)

    twostim_rlf_df = pd.DataFrame(twostim_rlf)
    return twostim_rlf_df


def twostim_gamma_power(df, data_dir):
    twostim_gamma_power = []

    for session in iter_sessions(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for chan in channels:
                stim_spikes = np.array(
                    rec.stim_spiketrains(channel_number=chan, ignore_onset=0.050),
                    dtype="object",
                )
                spiketrains_by_level = group_by_param(stim_spikes, varying_levels)

                lfp_arr = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                lfp_arr = lfp_arr - np.mean(lfp_arr, axis=0)
                bandpass_lfp = bandpass_filter(
                    lfp_arr,
                    20,
                    50,
                    fs=lfp_samplingrate,
                    order=1,
                )

                trial_power = np.array(
                    [
                        get_power(
                            bandpass_lfp[trial_index],
                            baseline_start=trial_delays[trial_index]
                            - trial_durations[trial_index],
                            baseline_stop=trial_delays[trial_index] - 0.05,
                            stim_start=trial_delays[trial_index] + 0.05,
                            stim_stop=trial_delays[trial_index]
                            + trial_durations[trial_index],
                            samplingrate=lfp_samplingrate,
                        )
                        for trial_index in rec.powlfile.trial_indexes
                    ]
                )
                power_by_level = group_by_param(trial_power, varying_levels)
                lfp_by_level = group_by_param(bandpass_lfp, varying_levels)

                for level, level_power in power_by_level.items():
                    relative_level = level - fixed_level
                    level_phaselocking = get_phaselocking(
                        lfp_by_level[level], spiketrains_by_level[level]
                    )
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "gammapower": np.mean(level_power),
                        "gammapower_sem": scipy.stats.sem(level_power),
                        "gamma_plv": level_phaselocking["vector_strength"],
                        "gamma_plv_angle": level_phaselocking["mean_phase"],
                        "gamma_plv_p": level_phaselocking["p"],
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                    }

                    twostim_gamma_power.append(tmp)

    twostim_gamma_power_df = pd.DataFrame(twostim_gamma_power)
    return twostim_gamma_power_df


if __name__ == "__main__":
    raise SystemExit(main())
