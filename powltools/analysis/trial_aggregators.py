from typing import Any, Callable

TrialParams = dict[str, Any]
TrialParamFunc = Callable[[TrialParams], Any]
