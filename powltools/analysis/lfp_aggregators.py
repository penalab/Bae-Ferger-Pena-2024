from typing import Any, Callable
import numpy as np
import numpy.typing as npt

LFPType = npt.NDArray[np.float_]
LFPSnippetsType = dict[int, LFPType]
LFPSnippetFunc = Callable[[LFPType], Any]
