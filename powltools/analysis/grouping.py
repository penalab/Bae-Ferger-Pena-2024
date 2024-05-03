import numpy as np
import numpy.typing as npt


def group_by_param(
    trial_values: npt.ArrayLike,
    trial_params: npt.ArrayLike,
) -> dict[float, np.ndarray]:
    """Group any values by common unique parameters

    For example, group response rates by stimulus level:

        # Input arrays:
        responses = rec.response_rates(channel_number=1, stimulus_index=0)
        levels = rec.aggregate_stim_params(stim_level, stimulus_index=0)
        # Grouping into a dict:
        responses_by_level = group_by_param(responses, levels)
        # For a rate level function with errorbars:
        plot_levels, resp_mean, resp_std = np.array(
            [
                [level, resp.mean(), resp.std()]
                for level, resp in responses_by_level.items()
            ]
        ).T
        plt.errorbar(plot_levels, resp_mean, yerr=resp_std)

    """
    trial_values = np.asarray(trial_values)
    trial_params = np.asarray(trial_params)
    uparameters = np.unique(trial_params, axis=0)
    if trial_params.ndim != 1:
        raise ValueError(
            "Use `group_by_multiparam()` for grouping by multiple parameters."
        )
    return {p: trial_values[trial_params == p] for p in uparameters}


def group_by_multiparam(
    trial_values: npt.ArrayLike,
    trial_params: npt.ArrayLike,
) -> dict[tuple[float, ...], np.ndarray]:
    """Group values by a set of parameters, such as the azi/ele position

    Similar to group_by_param, but the keys of the returned dict are tuples
    with the combination of parameters.

    Example of a color-coded spatial receptive field:

        # Input arrays:
        responses = rec.response_rates(channel_number=1, stimulus_index=0)
        positions = rec.aggregate_stim_params(stim_position, stimulus_index=0)
        # Grouping into a dict:
        responses_by_position = group_by_param(responses, positions)
        # For a spatial receptive field scatterplot:
        azimuths, elevations, resp_mean = np.array(
            [
                [azi, ele, resp.mean()]
                for (azi, ele), resp in responses_by_position.items()
            ]
        ).T
        plt.scatterplot(azimuths, elevations, c=resp_mean)

    """
    trial_values = np.asarray(trial_values)
    trial_params = np.asarray(trial_params)
    uparameters = np.unique(trial_params, axis=0)
    return {
        tuple(p): trial_values[(trial_params == p).all(axis=1)] for p in uparameters
    }
