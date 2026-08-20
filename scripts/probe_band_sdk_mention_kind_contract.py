"""Prove the declared band-sdk floor really puts a mention ``kind`` on the wire.

The plugin demotes a self @mention to ``kind="reference"`` because Band answers a
self *delivery* mention with ``422 cannot_mention_self`` and rejects the whole
message. That fix is only real if ``kind`` survives serialisation — and it rides
on thin ice: the generated ``ChatMessageRequestMentionsItem`` declares only
``handle``/``id``/``name``, so ``kind`` is carried purely because the model's
pydantic config is ``extra="allow"``. A regen that flips that to ``"forbid"``, or
a Fern release that stops serialising extras, would silently drop the field — and
an absent ``kind`` **defaults back to ``"mention"``**, which is exactly the 422
the demotion exists to avoid.

Silent is the problem: nothing in the plugin's own suite can see it, because the
unit tests assert on the object we constructed, not on the bytes. Hence a probe,
and hence one that deliberately imports nothing from this package.
"""

from importlib.metadata import version

from band.client.rest import ChatMessageRequestMentionsItem
from band_rest.core.jsonable_encoder import jsonable_encoder


def main() -> None:
    assert version("band-sdk") == "1.3.0"

    # The field is undeclared, so this is the assertion that matters: the model
    # must keep an unexpected kwarg rather than reject or discard it.
    item = ChatMessageRequestMentionsItem(
        id="00000000-0000-0000-0000-000000000000",
        handle="owner/agent",
        kind="reference",
    )
    assert item.kind == "reference"

    encoded = jsonable_encoder(item)
    assert encoded["kind"] == "reference", (
        f"band-sdk {version('band-sdk')} dropped the mention kind on the wire: "
        f"{encoded!r} — a mention without an explicit kind defaults to "
        f'"mention", so a self reference would become a self mention and Band '
        f"would reject the whole message with 422 cannot_mention_self"
    )

    # An omitted kind must stay omitted: the server default is what every normal
    # recipient relies on, and sending an explicit null is rejected outright.
    plain = jsonable_encoder(
        ChatMessageRequestMentionsItem(id="00000000-0000-0000-0000-000000000001")
    )
    assert "kind" not in plain, f"unexpected kind on a plain mention: {plain!r}"

    print("band-sdk 1.3.0 mention-kind contract: OK")


if __name__ == "__main__":
    main()
