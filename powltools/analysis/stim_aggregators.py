from typing import Any, Callable

StimulusParams = dict[str, Any]
StimParamFunc = Callable[[StimulusParams], Any]


def stim_position(stim_params: StimulusParams) -> tuple[int, int]:
    return (stim_params["azi"], stim_params["ele"])


def stim_level(stim_params: StimulusParams) -> float:
    return stim_params["stim_func_kwargs"]["level"]


def stim_delay(stim_params: StimulusParams) -> float:
    return stim_params["delay_s"]


def stim_len(stim_params: StimulusParams) -> float:
    return stim_params["stim_func_factory_kwargs"]["stim_len_s"]


def stim_f_am(stim_params: StimulusParams) -> int:
    return stim_params["stim_func_factory_kwargs"]["modulation_frequency"]
