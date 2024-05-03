# Builtin
from itertools import combinations
import pathlib
import glob
from typing import cast
from functools import cache

# 3rd party
import numpy as np
import scipy
import pandas as pd

# Custom
from powltools.io.file import POwlFile
from powltools.analysis.recording import Recording
from powltools.analysis.stim_aggregators import stim_position
from powltools.analysis.grouping import group_by_param
from settings_rawdata_path import datadir_ot, units_ot, srf_dates_ot
from settings_rawdata_path import datadir_icx, units_icx, srf_dates_icx
from analyze_flat_noise import filter_relative_level_flat
from funcs_common import iter_sessions


def main():
    # Single stimulus data, flat noise, curated/clean
    # Contains one line per [date, owl, channel, level]
    OUTDIR = pathlib.Path("./intermediate_results").absolute()
    OUTDIR.mkdir(exist_ok=True)
    for region in ["ot", "icx"]:
        if region == "ot":
            data_dir = datadir_ot
            df = pd.read_csv(units_ot)
            df_srf = pd.read_csv(srf_dates_ot)
        elif region == "icx":
            data_dir = datadir_icx
            df = pd.read_csv(units_icx)
            df_srf = pd.read_csv(srf_dates_icx)
        df_srf = df_srf.set_index("competition")
        df = df[df["owl"] == 40]

        df = df.join(df_srf, on="date", how="inner")
        print(f"{df.shape = }")
        df.set_index(["date", "owl"], inplace=True)

        # Find the DRIVER azimuth and elevation for each session:
        df["azimuth"] = pd.Series(dtype=float)
        df["elevation"] = pd.Series(dtype=float)
        for session in iter_sessions(
            df[["channel"]], filter_relative_level_flat, data_dir
        ):
            rec = Recording(POwlFile(session["filenames"][-1]))
            df.loc[(session["date"], session["owl"]), "azimuth"] = cast(
                float, rec.stimuli_parameters(stimulus_index=0)[0]["azi"]
            )
            df.loc[(session["date"], session["owl"]), "elevation"] = cast(
                float, rec.stimuli_parameters(stimulus_index=0)[0]["ele"]
            )
        df.set_index("channel", append=True, inplace=True)
        df.sort_index(inplace=True)

        print("population_elevation_tunings".upper())
        tunings_df = population_elevation_tunings(df, data_dir)
        tunings_df.to_feather(OUTDIR / f"elevation_tunings_{region}.feather")

        print("population_elevation_signalcorr".upper())
        sigcorr_df = population_elevation_signalcorr(df, data_dir)
        sigcorr_df.to_feather(OUTDIR / f"elevation_signalcorr_{region}.feather")

    return 0


def population_elevation_tunings(df, data_dir: pathlib.Path) -> pd.DataFrame:
    df2 = df.reset_index()
    df2["best_ele"] = pd.Series(dtype=float)
    df2["ele_width"] = pd.Series(dtype=float)

    for index, row in df2.iterrows():
        rec = get_srf_rec(row["srf"], row["owl"], data_dir)
        x, y = elevation_tuning(rec, row["channel"], row["azimuth"])
        fit_res = fit_elevation_tuning(x, y)
        df2.loc[index, "best_ele"] = fit_res["ele_best"]
        df2.loc[index, "ele_width"] = fit_res["ele_width"]
    return df2


def population_elevation_signalcorr(df, data_dir: pathlib.Path) -> pd.DataFrame:
    df3 = df.reset_index().set_index(["date", "azimuth", "owl"])

    sig_corr_out = []

    for index in np.unique(df3.index):
        idx_date, idx_azimuth, idx_owl = index

        rec = get_srf_rec(df3.loc[index, "srf"].iloc[0], idx_owl, data_dir)

        channels = np.unique(df3.loc[index, "channel"])
        for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
            signal_corrcoeff = elevation_signal_correlation(
                rec, chan1, chan2, idx_azimuth
            )

            tmp = {
                "date": idx_date,
                "owl": idx_owl,
                "azimuth": idx_azimuth,
                "elevation": df3.loc[index, "elevation"].iloc[0],
                "channel1": chan1,
                "channel2": chan2,
                "srf_recording": df3.loc[index, "srf"].iloc[0],
                "signal_corr": signal_corrcoeff,
            }

            sig_corr_out.append(tmp)

    return pd.DataFrame(sig_corr_out)


@cache
def get_srf_rec(datestr: str, owl: int | str, data_dir: pathlib.Path) -> Recording:
    filepath = (
        data_dir
        / f"{datestr.replace('-', '')}_{owl}_awake"
        / "[0-9][0-9]_spatial_receptive_field.h5"
    ).absolute()
    filepath = glob.glob(str(filepath))[0]
    # print(filepath)
    return Recording(POwlFile(filepath))


def elevation_tuning(rec: Recording, channel_number: int, azimuth: int):
    azimuths, elevations = np.array(rec.aggregate_stim_params(stim_position)).T
    responses = np.array(
        rec.response_rates(channel_number=channel_number, stimulus_index=0)
    )

    mask = azimuths == azimuth
    azimuths = azimuths[mask]
    elevations = elevations[mask]
    responses = responses[mask]
    ele_responses = group_by_param(responses, elevations)
    tuning_elevations, tuning_responses = np.array(
        [(ele, np.mean(resp)) for ele, resp in sorted(ele_responses.items())]
    ).T
    return tuning_elevations, tuning_responses


def elevation_signal_correlation(rec: Recording, channel1: int, channel2, azimuth: int):
    tuning_elevations1, tuning_responses1 = elevation_tuning(rec, channel1, azimuth)
    tuning_elevations2, tuning_responses2 = elevation_tuning(rec, channel2, azimuth)
    if not np.array_equal(tuning_elevations1, tuning_elevations2):
        raise ValueError("Not the same elevations!")

    return np.corrcoef(tuning_responses1, tuning_responses2)[0, 1]


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((np.asarray(x) - mu) / sigma) ** 2)


def make_error_function(elevations, responses):
    elevations = np.asarray(elevations)
    responses = np.asarray(responses)
    responses = responses / np.max(responses)

    def error_fun(mu, sigma):
        return np.sqrt(np.mean((gaussian(elevations, mu, sigma) - responses) ** 2))

    return lambda mu_sigma: error_fun(*mu_sigma)


def fit_elevation_tuning(elevations, responses):
    # Normalize responses:
    responses = responses - responses.min()
    responses /= responses.max()

    error_function = make_error_function(elevations, responses)

    res = scipy.optimize.minimize(
        fun=error_function,
        x0=np.array([-10.0, 25.0]),
        bounds=[(-90.0, +90.0), (10.0, 180.0)],
    )
    return {
        "ele_best": res.x[0],
        "ele_width": res.x[1] * 2,
        "fit_results": res,
        "norm_responses": responses,
    }


if __name__ == "__main__":
    raise SystemExit(main())
