from __future__ import annotations as _annotations
from functools import cached_property
import os
import re

import numpy as np
from typing import (
    Any,
    Iterable,
    Literal,
    NotRequired,
    TypedDict,
    Unpack,
    cast,
)
import h5py

from .trial_aggregators import TrialParams, TrialParamFunc
from .stim_aggregators import StimParamFunc, StimulusParams
from .stim_aggregators import stim_delay, stim_len
from .spike_aggregators import SpikeTimesType, SpikeTrainsType, SpikeTimesFunc
from .spike_aggregators import spike_count_response
from .lfp_aggregators import LFPSnippetsType, LFPSnippetFunc

from ..io.file import POwlFile
from ..io.parameters import get_params, GroupParams


class StimulusNotFound(KeyError):
    def __init__(self, recording: Recording, stimulus_index: int) -> None:
        super().__init__()
        self.stimulus_index = stimulus_index
        self.recording = recording

    def __repr__(self) -> str:
        return f"StimulusNotFound(recording={self.recording!r}, stimulus_index={self.stimulus_index!r})"

    def __str__(self) -> str:
        return f"StimulusNotFound: Stimulus with index {self.stimulus_index} not found for {self.recording!r}."


class Recording:
    """Representation of one (Batch) Recording

    This class provides some low- to mid-level caching and mechanisms to
    aggregate data across trials from one bach recording.

    The assumption is that the provided POwlFile contains processed data
    including especially the "spiketrains".

    See also
    --------
    AdhocRecording : Calculates spiketains dynamically from unfiltered traces.
    """

    def __init__(self, powlfile: POwlFile):
        self.powlfile = powlfile
        self._cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(POwlFile({self.powlfile.filepath!r}))"

    def __str__(self) -> str:
        return f"{self!r}"

    def global_parameters(self) -> GlobalParamsTypeV0 | GlobalParamsType:
        cache_key = "global_parameters"
        if cache_key not in self._cache:
            with self.powlfile as file:
                self._cache[cache_key] = get_params(file)
        return self._cache[cache_key]

    @cached_property
    def powl_version(self) -> str:
        return self.global_parameters().get("powl_version", "0.0.0")

    def get_samplingrate(self, which: str) -> float:
        if self.powl_version.startswith("0"):
            global_params = cast(GlobalParamsTypeV0, self.global_parameters())
            if which in global_params:
                return global_params[which]
            elif which in ("traces",):
                return global_params["adc_samplingrate"]
            elif which in ("stimuli",):
                return global_params["dac_samplingrate"]
        else:
            global_params = cast(GlobalParamsType, self.global_parameters())
            samplingrates = global_params["samplingrates"]
            if which in samplingrates:
                return samplingrates[which]
            elif which in ("adc_samplingrate",):
                return samplingrates["traces"]
            elif which in ("dac_samplingrate", "stimuli"):
                return samplingrates["stimuli_freefield"]
        raise ValueError(f"Cannot find samplingrate for {which=}")

    @cached_property
    def session_id(self):
        global_params = self.global_parameters()
        if "session" in global_params:
            owl = global_params["session"].get("owl", "unknown")
            date = global_params["session"].get("date", "unknown").replace("-", "")
            return f"{date}_{owl}"
        else:
            return (
                (self.powlfile.filepath.parent.name)
                .removesuffix("_awake")
                .removesuffix("_anesthetized")
            )

    def trials_parameters(self) -> dict[int, TrialParams]:
        cache_key = "trials_parameters"
        if cache_key not in self._cache:
            self._cache[cache_key] = {
                trial_index: get_params(trial_group)
                for trial_index, trial_group in self.powlfile.trials()
            }
        return self._cache[cache_key]

    def aggregate_trial_params(self, func: TrialParamFunc) -> list[Any]:
        params = self.trials_parameters()
        return [
            func(params[trial_index]) for trial_index in self.powlfile.trial_indexes
        ]

    def stimuli_parameters(self, stimulus_index=0) -> dict[int, StimulusParams]:
        cache_key = f"stimulus_parameters({stimulus_index})"
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = {
                    trial_index: get_params(
                        cast(h5py.Group, trial_group[f"stimuli/{stimulus_index}"])
                    )
                    for trial_index, trial_group in self.powlfile.trials()
                }
            except KeyError:
                raise StimulusNotFound(recording=self, stimulus_index=stimulus_index)
        return self._cache[cache_key]

    def channel_numbers(self) -> list[int]:
        return self.powlfile.channel_numbers("spiketrains")

    def aggregate_stim_params(
        self,
        func: StimParamFunc,
        stimulus_index: int = 0,
    ) -> list[Any]:
        params = self.stimuli_parameters(stimulus_index)
        return [
            func(params[trial_index]) for trial_index in self.powlfile.trial_indexes
        ]

    def spike_trains(self, channel_number: int) -> SpikeTrainsType:
        cache_key = f"spike_trains({channel_number})"
        if cache_key not in self._cache:
            spiketrain_dict: SpikeTrainsType = {}
            data_key = f"spiketrains/{channel_number}/data_array"
            for trial_index, trial_group in self.powlfile.trials():
                spiketrain_dict[trial_index] = np.asarray(trial_group[data_key])
            self._cache[cache_key] = spiketrain_dict
        return self._cache[cache_key]

    def aggregrate_spikes(
        self,
        func: SpikeTimesFunc,
        *arg_lists: Unpack[tuple[Iterable[Any], ...]],
        channel_number: int,
    ) -> list[Any]:
        spiketrains = self.spike_trains(channel_number=channel_number)
        return [
            func(spiketrains[trial_index], *args)
            for trial_index, *args in zip(self.powlfile.trial_indexes, *arg_lists)
        ]

    def response_rates(self, channel_number: int, stimulus_index: int) -> list[int]:
        return self.aggregrate_spikes(
            spike_count_response,
            self.aggregate_stim_params(stim_delay, stimulus_index=stimulus_index),
            self.aggregate_stim_params(stim_len, stimulus_index=stimulus_index),
            channel_number=channel_number,
        )

    def stim_spiketrains(
        self, channel_number: int, stimulus_index: int = 0, ignore_onset: float = 0.0
    ) -> list[SpikeTimesType]:
        _agg_spike_times = lambda st, delay, dur: st[
            np.searchsorted(st, delay + ignore_onset) : np.searchsorted(st, delay + dur)
        ]
        return self.aggregrate_spikes(
            _agg_spike_times,
            self.aggregate_stim_params(stim_delay, stimulus_index=stimulus_index),
            self.aggregate_stim_params(stim_len, stimulus_index=stimulus_index),
            channel_number=channel_number,
        )

    def lfp_snippets(self, channel_number: int) -> LFPSnippetsType:
        cache_key = f"lfp_snippets({channel_number})"
        if cache_key not in self._cache:
            lfp_dict: LFPSnippetsType = {}
            data_key = f"lfp/{channel_number}/data_array"
            for trial_index, trial_group in self.powlfile.trials():
                lfp_dict[trial_index] = np.asarray(trial_group[data_key])
            self._cache[cache_key] = lfp_dict
        return self._cache[cache_key]

    def aggregrate_lfps(
        self,
        func: LFPSnippetFunc,
        *arg_lists: * tuple[Iterable[Any], ...],
        channel_number: int,
    ) -> list[Any]:
        lfpsnippets = self.lfp_snippets(channel_number=channel_number)
        return [
            func(lfpsnippets[trial_index], *args)
            for trial_index, *args in zip(self.powlfile.trial_indexes, *arg_lists)
        ]

    @cached_property
    def fileinfo(self):
        m = re.match(
            r"(?P<index>[0-9]+)_(?P<paradigm>.*?)(?:_(?P<number>[0-9]+))?\.h5",
            self.filename,
        )
        if m is None:
            raise ValueError(f"Can not extract fileinfo from {self.filename!r}")
        return m.groupdict()

    @cached_property
    def filename(self):
        return os.path.basename(self.powlfile.filepath)


class SessionParamsType(TypedDict):
    condition: Literal["awake", "anesthetized"]
    date: str  # like '2023-04-11'
    hemisphere: Literal["left", "right"]
    owl: int


class SignalParamsType(TypedDict):
    samplingrate: float
    leading: NotRequired[int]
    trailing: NotRequired[int]


class GlobalParamsTypeV0(TypedDict):
    dac_samplingrate: float
    adc_samplingrate: float
    session: SessionParamsType
    regions: dict[int, str]
    signals: dict[str, SignalParamsType]


class GlobalParamsType(TypedDict):
    powl_version: str
    datetime: str  # like "2024-01-24 09:35:17"
    samplingrates: dict[str, float]
    owl_info: dict
    batch_spec: dict
    hardware_config: dict
    session: SessionParamsType
    regions: dict[int, str]
    signals: dict[str, SignalParamsType]
