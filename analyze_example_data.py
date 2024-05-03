# Builtin
import pathlib
from typing import cast

# 3rd party
import numpy as np
import pandas as pd

# Custom
from powltools.io.file import POwlFile
from powltools.analysis.recording import Recording
from powltools.analysis.stim_aggregators import stim_level, stim_position
from powltools.analysis.grouping import group_by_multiparam
from funcs_common import fixed_var_stimuli
from settings_rawdata_path import datadir_ot, datadir_icx


def main():
    OUTDIR = pathlib.Path("./intermediate_results").absolute()
    OUTDIR.mkdir(exist_ok=True)
    # SRF data (Figure 1 A):
    df = srf_plot_data(
        filename=datadir_ot / "20230420_33_awake/01_spatial_receptive_field.h5",
        channel_number=8,
    )
    df.to_feather(OUTDIR / "example_srf_20230420_33.feather")

    # Coincident Plot Data
    # Figure 2 OT (Flat)
    df = conincident_plot_data(
        filename=datadir_ot / "20230523_40_awake/02_relative_intensity.h5",
        channel_number1=3,
        channel_number2=8,
    )
    df.to_feather(OUTDIR / "example_spiketrains_ot_flat_20230523_40.feather")
    # Figure 3C OT (Driver 55 Hz, Competitor 75 Hz)
    df = conincident_plot_data(
        filename=datadir_ot / "20230523_40_awake/04_amplitude_modulation_2.h5",
        channel_number1=3,
        channel_number2=8,
    )
    df.to_feather(OUTDIR / "example_spiketrains_ot_driver55_20230523_40.feather")
    # Figure 3D OT (Driver 75 Hz, Competitor 55 Hz)
    df = conincident_plot_data(
        filename=datadir_ot / "20230523_40_awake/03_amplitude_modulation_1.h5",
        channel_number1=3,
        channel_number2=8,
    )
    df.to_feather(OUTDIR / "example_spiketrains_ot_driver75_20230523_40.feather")

    # Figure 5A ICx (Flat)
    df = conincident_plot_data(
        filename=datadir_icx / "20230730_40_awake/04_relative_intensity.h5",
        channel_number1=5,
        channel_number2=9,
    )
    df.to_feather(OUTDIR / "example_spiketrains_icx_flat_20230523_40.feather")
    # Figure 6A ICx (Driver 55 Hz, Competitor 75 Hz)
    df = conincident_plot_data(
        filename=datadir_icx / "20230731_40_awake/03_amplitude_modulation_2.h5",
        channel_number1=5,
        channel_number2=9,
    )
    df.to_feather(OUTDIR / "example_spiketrains_icx_driver55_20230523_40.feather")
    # Figure 6B ICx (Driver 75 Hz, Competitor 55 Hz)
    df = conincident_plot_data(
        filename=datadir_icx / "20230731_40_awake/02_amplitude_modulation_1.h5",
        channel_number1=5,
        channel_number2=9,
    )
    df.to_feather(OUTDIR / "example_spiketrains_icx_driver75_20230523_40.feather")


def srf_plot_data(
    filename: pathlib.Path,
    channel_number: int,
):
    rec = Recording(POwlFile(str(filename)))
    positions = rec.aggregate_stim_params(stim_position)
    trial_responses = rec.response_rates(
        channel_number=channel_number, stimulus_index=0
    )
    df = pd.DataFrame(
        [
            {
                "channel_number": channel_number,
                "azimuth": azi,
                "elevation": ele,
                "response_mean": resp.mean(),
                "response_std": resp.std(),
                "n_trials": resp.size,
            }
            for (azi, ele), resp in group_by_multiparam(
                trial_responses, positions
            ).items()
        ]
    )
    return df


def conincident_plot_data(
    filename: pathlib.Path,
    channel_number1: int,
    channel_number2: int,
):
    rec = Recording(POwlFile(str(filename)))
    fixed_varying = fixed_var_stimuli(rec, stim_level)
    fixed_level = cast(float, fixed_varying["fixed_value"])
    varying_levels = np.array(
        rec.aggregate_stim_params(
            stim_level, stimulus_index=fixed_varying["varying_index"]
        )
    )
    trial_relative_levels = varying_levels - fixed_level
    df = pd.DataFrame(
        {
            "driver_level": fixed_level,
            "competitor_level": varying_levels,
            "relative_level": trial_relative_levels,
            "channel_unit1": channel_number1,
            "channel_unit2": channel_number2,
            "spiketrain_unit1": rec.stim_spiketrains(
                channel_number=channel_number1, ignore_onset=0.000
            ),
            "spiketrain_unit2": rec.stim_spiketrains(
                channel_number=channel_number2, ignore_onset=0.000
            ),
        }
    )
    return df


if __name__ == "__main__":
    main()
