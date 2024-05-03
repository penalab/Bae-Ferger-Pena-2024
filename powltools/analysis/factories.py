from pathlib import Path
import h5py
from typing import cast
from ..io.file import POwlFile
from .recording import Recording
from .adhoc_recording import AdhocRecording


def make_recording(powlfile_path: str | Path):
    """Create either a Recording or AdhocRecording instance"""

    pf = POwlFile(powlfile_path)
    with pf as file:
        has_spiketrains = "spiketrains" in cast(h5py.Group, file["/trials/0"])
    if has_spiketrains:
        return Recording(pf)
    else:
        return AdhocRecording(pf)
