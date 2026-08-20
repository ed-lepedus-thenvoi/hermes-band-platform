"""Outbound mention policy: who is addressed, and how it renders.

No live Band credentials are used. The small renderer below models the
externally observed behavior: a matching display handle is replaced in place;
when no safe match exists, the structured mention is placed separately. Tests
assert rendered message content, not implementation details.
"""

import pytest

from hermes_band_platform.adapter import (
    _mention_items,
    _substitutes_safely,
    align_mentions_to_content,
)


def _simulate_server(content, mention_items):
    """Model the observable rendering behavior used by these tests.

    Matching display text is substituted in place; otherwise the structured
    mention is placed separately. The intentionally simple model exposes the
    longer-token corruption the adapter must prevent.
    """
    for item in mention_items:
        marker = f"@[[{item.id}]]"
        if marker in content:
            continue
        if item.handle and f"@{item.handle}" in content:
            content = content.replace(f"@{item.handle}", marker)
        elif item.name and f"@{item.name}" in content:
            content = content.replace(f"@{item.name}", marker)
        else:
            content = f"{marker} {content}"
    return content


def _mentions(*participants):
    return _mention_items(list(participants), agent_id="self")


class TestSubstitutionSafety:
    @pytest.mark.parametrize(
        "content,token",
        [
            ("@alice please look", "alice"),           # whole token, start
            ("cc @alice and @bob", "alice"),           # whole token, mid-sentence
            ("thanks @alice.", "alice"),               # trailing sentence period
            ("no mention of them here", "alice"),      # absent entirely
            ("@alice @alice again", "alice"),          # every occurrence whole
            ("@twins-owner/dumpty hi", "twins-owner/dumpty"),  # qualified handle
        ],
    )
    def test_safe_cases(self, content, token):
        assert _substitutes_safely(content, token) is True

    @pytest.mark.parametrize(
        "content,token",
        [
            ("@ageofascension/ted please look", "ageofascension"),  # #49
            ("ping @tedx", "ted"),                     # longer word
            ("mail @ed.lepedus today", "ed"),          # dotted handle
            ("see https://x/@alice for this", "alice"),  # inside a URL path
            ("@alice and @alice-bot", "alice"),        # one whole, one embedded
        ],
    )
    def test_unsafe_cases(self, content, token):
        assert _substitutes_safely(content, token) is False

    def test_matching_is_case_sensitive_like_the_server(self):
        # Matching is case-sensitive, so an "@Alice" in content is not a
        # match for the handle "alice" and cannot be corrupted by it.
        assert _substitutes_safely("@Alice/x hello", "alice") is True


class TestAlignment:
    def test_the_49_case_no_longer_corrupts_the_content(self):
        """The production bug: the owner's handle sits inside a longer handle."""
        content = "@ageofascension/ted please take a look"
        mentions = _mentions(
            {"id": "owner-uuid", "type": "User", "handle": "ageofascension",
             "name": "Ed Lepedus"}
        )

        aligned = align_mentions_to_content(content, mentions)
        rendered = _simulate_server(content, aligned)

        # Before: "@[[owner-uuid]]/ted please take a look" -> "@Ed Lepedus/ted".
        assert rendered == "@[[owner-uuid]] @ageofascension/ted please take a look"
        assert aligned[0].handle is None
        assert aligned[0].id == "owner-uuid"

    def test_withholding_a_field_keeps_the_mention_kind(self):
        """A demoted self mention must not be re-promoted by the rebuild.

        Alignment only rebuilds an item when it withholds a field, so a dropped
        ``kind`` would revert the entry to the server's ``"mention"`` default
        exactly and only on the corrupting-handle path — and Band answers a self
        delivery mention with 422 cannot_mention_self for the whole message.
        """
        content = "@ageofascension/ted please take a look"
        mentions = _mention_items(
            [
                {
                    "id": "agent-self",
                    "type": "Agent",
                    "handle": "ageofascension",
                    "name": "BandAId",
                }
            ],
            agent_id="agent-self",
            explicit_ids=["agent-self"],
        )
        assert mentions[0].kind == "reference"

        aligned = align_mentions_to_content(content, mentions)

        assert aligned[0].handle is None        # withheld -> the item was rebuilt
        assert aligned[0].kind == "reference"   # and the kind survived it

    def test_a_matching_handle_is_left_alone_so_it_renders_in_place(self):
        """The case #51's strip broke: substitution here is the good outcome."""
        content = "@twins-owner/dumpty please reply with ACK"
        mentions = _mentions(
            {"id": "6629", "type": "User", "handle": "twins-owner/dumpty",
             "name": "Dumpty"}
        )

        aligned = align_mentions_to_content(content, mentions)
        rendered = _simulate_server(content, aligned)

        assert aligned[0].handle == "twins-owner/dumpty"
        assert rendered == "@[[6629]] please reply with ACK"

    def test_placement_is_preserved_mid_sentence(self):
        content = "could @alice and @bob both confirm?"
        mentions = _mentions(
            {"id": "a", "type": "User", "handle": "alice"},
            {"id": "b", "type": "User", "handle": "bob"},
        )

        rendered = _simulate_server(
            content, align_mentions_to_content(content, mentions)
        )

        assert rendered == "could @[[a]] and @[[b]] both confirm?"

    def test_an_unrelated_handle_in_prose_is_untouched(self):
        content = "ask @other/person about it"
        mentions = _mentions({"id": "u", "type": "User", "handle": "alice"})

        rendered = _simulate_server(
            content, align_mentions_to_content(content, mentions)
        )

        assert rendered == "@[[u]] ask @other/person about it"

    @pytest.mark.parametrize(
        "content,handle,description",
        [
            ("mail alice@example.com about it", "example.com", "e-mail domain"),
            ("write to @alice@example.com", "alice", "handle-shaped e-mail local part"),
            ("see https://git.io/@alice/repo", "alice", "URL path segment"),
            ("run `curl -H '@bob: x'` first", "alice", "unrelated handle in code"),
            ('he said "@bob will do it"', "alice", "unrelated quoted handle"),
            ("the @alice-bot service is down", "alice", "hyphenated service name"),
        ],
    )
    def test_controls_content_survives_untouched(self, content, handle, description):
        """Nothing but a real, whole-token mention may be rewritten."""
        mentions = _mentions({"id": "u", "type": "User", "handle": handle})

        rendered = _simulate_server(
            content, align_mentions_to_content(content, mentions)
        )

        assert rendered == f"@[[u]] {content}", description

    @pytest.mark.parametrize(
        "content",
        ["run `curl -H '@alice: x'` first", 'he said "@alice will do it"'],
    )
    def test_known_limit_a_real_mention_inside_code_or_quotes_is_substituted(
        self, content
    ):
        """Documented limitation, not an oversight.

        When the mentioned participant's own handle appears inside a code span
        or a quotation, the server substitutes it there like anywhere else. The
        adapter does not parse markdown, and guessing at code spans would be
        both fragile and unable to help — the server would still rewrite what it
        found. The cost is cosmetic (a resolved name inside a code span), not
        corruption, which is what this module exists to prevent.

        Unrelated handles in the same positions are untouched; that is the case
        that matters and it is covered above.
        """
        mentions = _mentions({"id": "u", "type": "User", "handle": "alice"})

        rendered = _simulate_server(
            content, align_mentions_to_content(content, mentions)
        )

        assert "@[[u]]" in rendered
        assert "@alice" not in rendered

    def test_an_unsafe_name_is_withheld_too(self):
        """The server falls through to @name when the handle does not match."""
        content = "@Ed Lepeduson wrote this"
        mentions = _mentions(
            {"id": "u", "type": "User", "handle": "someone-else", "name": "Ed Lepedus"}
        )

        aligned = align_mentions_to_content(content, mentions)

        assert aligned[0].name is None
        assert _simulate_server(content, aligned).startswith("@[[u]] ")

    def test_a_missing_handle_is_not_invented(self):
        content = "please review"
        mentions = _mentions({"id": "u", "type": "User"})

        aligned = align_mentions_to_content(content, mentions)

        assert aligned[0].handle is None
        assert _simulate_server(content, aligned) == "@[[u]] please review"

    def test_content_is_never_modified(self):
        content = "@ageofascension/ted look at @tedx too"
        mentions = _mentions(
            {"id": "o", "type": "User", "handle": "ageofascension"},
            {"id": "t", "type": "Agent", "handle": "ted"},
        )

        align_mentions_to_content(content, mentions)

        assert content == "@ageofascension/ted look at @tedx too"

    def test_empty_input_is_tolerated(self):
        assert align_mentions_to_content("hi", []) == []
        assert align_mentions_to_content("", None) == []
