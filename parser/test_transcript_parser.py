"""
Unit tests for the transcript parser.
"""

import pytest
from transcript_parser import (
    _parse_speaker_turns,
    _extract_qa,
)


# ---------- Fixtures ----------

SAMPLE_SECTION_NO_AUDIENCE = """WARREN BUFFETT: I'd like to introduce our managers.
Charlie, do you have anything to add?
CHARLIE MUNGER: No."""

SAMPLE_SECTION_NO_MUNGER = """AUDIENCE MEMBER: What do you think about tech stocks?
WARREN BUFFETT: We don't invest in things we don't understand.
We stay within our circle of competence."""

SAMPLE_SECTION_FULL_QA = """AUDIENCE MEMBER: Why don't you split the stock?
WARREN BUFFETT: We don't think splitting makes sense for our shareholder base.
We want to attract long-term investors.
CHARLIE MUNGER: I think the idea of carving ownership into tiny pieces is almost insane.
And it's quite inefficient to service small accounts."""

SAMPLE_MULTI_TURN_QA = """AUDIENCE MEMBER: How do you value a business?
WARREN BUFFETT: We look at the present value of future cash flows.
CHARLIE MUNGER: Yes, it's all about the cash.
WARREN BUFFETT: Charlie is right. And we want a margin of safety.
CHARLIE MUNGER: A big margin of safety."""


# ---------- Tests ----------


class TestParseSpeakerTurns:
    def test_identifies_speakers(self):
        turns = _parse_speaker_turns(SAMPLE_SECTION_FULL_QA)
        speakers = [s for s, _ in turns]
        assert "AUDIENCE MEMBER" in speakers
        assert "WARREN BUFFETT" in speakers
        assert "CHARLIE MUNGER" in speakers

    def test_captures_speech_text(self):
        turns = _parse_speaker_turns(SAMPLE_SECTION_FULL_QA)
        buffett_turns = [(s, t) for s, t in turns if s == "WARREN BUFFETT"]
        assert len(buffett_turns) == 1
        assert "splitting" in buffett_turns[0][1]

    def test_multi_turn_conversation(self):
        turns = _parse_speaker_turns(SAMPLE_MULTI_TURN_QA)
        assert len(turns) == 5
        assert turns[0][0] == "AUDIENCE MEMBER"
        assert turns[1][0] == "WARREN BUFFETT"
        assert turns[2][0] == "CHARLIE MUNGER"
        assert turns[3][0] == "WARREN BUFFETT"
        assert turns[4][0] == "CHARLIE MUNGER"

    def test_empty_text(self):
        turns = _parse_speaker_turns("")
        assert turns == []


class TestExtractQA:
    def test_full_qa_returns_both_answers(self):
        result = _extract_qa(SAMPLE_SECTION_FULL_QA)
        assert result is not None
        buffett, munger = result
        assert "split" in buffett.lower()
        assert "insane" in munger.lower()

    def test_no_audience_returns_none(self):
        result = _extract_qa(SAMPLE_SECTION_NO_AUDIENCE)
        assert result is None

    def test_no_munger_returns_none(self):
        result = _extract_qa(SAMPLE_SECTION_NO_MUNGER)
        assert result is None

    def test_empty_text_returns_none(self):
        result = _extract_qa("")
        assert result is None

    def test_multi_turn_concatenates_answers(self):
        result = _extract_qa(SAMPLE_MULTI_TURN_QA)
        assert result is not None
        buffett, munger = result
        # Both Buffett turns should be concatenated
        assert "present value" in buffett
        assert "margin of safety" in buffett
        # Both Munger turns should be concatenated
        assert "cash" in munger
        assert "big margin" in munger
