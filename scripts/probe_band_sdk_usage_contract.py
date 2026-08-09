"""Isolated smoke test for the first real band-sdk usage-contract release."""

from importlib.metadata import version

from band.client.rest import ChatEventRequest
from band.core.types import USAGE_EVENT_TYPE, USAGE_METADATA_KEY, TurnUsage

import hermes_band_platform.usage_events as usage_events


def main() -> None:
    assert version("band-sdk") == "1.3.0"
    usage = TurnUsage.from_mapping(
        {"input": 7, "output": 3}, input="input", output="output"
    )
    assert (usage + usage).to_dict() == {
        "input_tokens": 14,
        "output_tokens": 6,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    event = ChatEventRequest(
        content="usage contract probe",
        message_type=USAGE_EVENT_TYPE,
        metadata={USAGE_METADATA_KEY: usage.to_dict()},
    )
    assert event.message_type == USAGE_EVENT_TYPE
    assert event.metadata[USAGE_METADATA_KEY] == usage.to_dict()
    assert usage_events._ensure_sdk_bindings() is True
    assert usage_events.TurnUsage is TurnUsage
    assert usage_events.USAGE_EVENT_TYPE == USAGE_EVENT_TYPE
    assert usage_events.USAGE_METADATA_KEY == USAGE_METADATA_KEY
    print("band-sdk 1.3.0 usage contract: OK")


if __name__ == "__main__":
    main()
