"""Mappings of electrode channels to their position or other relevant numbers.

`channel_number` always refers to the TDT/pOwl channel numbers.
"""

from __future__ import annotations
from .recording import Recording

ChannelNumber = int


def channel2region(rec: Recording, channel_number: ChannelNumber) -> str:
    electrodes: list[dict] = rec.global_parameters()["owl_info"]["electrodes"]  # type: ignore
    for electrode in electrodes:
        lowest, highest = [int(c) for c in electrode["channels"].split("-")]
        if lowest <= channel_number <= highest:
            return electrode["region"]
    return "region unknown"


class NeuroNexus16Channel:
    """NeuroNexus Electrodes A1x16 with 16 channels and H16 connector

    This maps TDT `channel_number`s as used in powl and powltools to their
    respective channel numbers by NeuroNexus.

    It also provides a function to sort `channel_number`s by the position
    along the shaft (from tip to base).
    """

    _POSITIONS_FROM_TIP: dict[ChannelNumber, int] = {
        channel_number: i
        for i, channel_number in enumerate(
            (12, 13, 10, 15, 9, 16, 5, 4, 8, 1, 7, 2, 6, 3, 11, 14)
        )
    }
    _NEURO_NEXUS_CHANNEL: dict[ChannelNumber, int] = {
        12: 1,
        16: 16,
        10: 2,
        15: 15,
        9: 3,
        16: 14,
        5: 4,
        4: 13,
        8: 5,
        1: 12,
        7: 6,
        2: 11,
        6: 7,
        3: 10,
        11: 8,
        14: 9,
    }

    @classmethod
    def sort_deepest_first(cls, channel_number: ChannelNumber) -> int:
        """Helper function to use for sorting channels based on their position.

        Usage example:
        from powltools.analysis.electrode_mapping import NeuroNexus16Channel as NN16
        channel_numbers = list(range(1,17))
        channels_top_to_bottom = sorted(channel_numbers, key=NN16.sort_deepest_first, reverse=True)
        """
        if channel_number > len(cls._POSITIONS_FROM_TIP):
            channel_number = (channel_number - 1) % len(cls._POSITIONS_FROM_TIP) + 1
        return cls._POSITIONS_FROM_TIP[channel_number]
