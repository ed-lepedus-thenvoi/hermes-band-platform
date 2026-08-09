"""Exercise the installed Band SDK activity contract without importing the plugin."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import os
from unittest.mock import AsyncMock

from band.platform.link import BandLink


async def _check_report_activity() -> None:
    link = BandLink("agent", "key", "wss://host/ws", "https://host")
    report = AsyncMock()
    link.rest.agent_api_activity.report_agent_chat_activity = report

    assert await link.report_activity("room-1", True) is True
    report.assert_awaited_once_with(
        chat_id="room-1",
        working=True,
        request_options={"timeout_in_seconds": 2, "max_retries": 0},
    )


def main() -> None:
    version = importlib.metadata.version("band-sdk")
    expected = os.environ.get("EXPECTED_BAND_SDK_VERSION")
    if expected:
        assert version == expected, (version, expected)

    signature = inspect.signature(BandLink.report_activity)
    assert list(signature.parameters) == [
        "self",
        "room_id",
        "working",
        "timeout_seconds",
    ]

    asyncio.run(_check_report_activity())
    print(f"band-sdk {version} activity contract: OK")


if __name__ == "__main__":
    main()
