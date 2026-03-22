"""
Unit tests for LangGraph node functions and routing logic.

All Gemini calls are mocked — no network or DB required.
Memory loading is mocked to return empty context.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import Gender, LifeStage, MessageType, Pet, Species, TriageLevel
from src.llm.graph.nodes import (
    finalize_node,
    generate_response_node,
    resolve_triage_node,
    rule_triage_node,
    should_override,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _user():
    u = MagicMock()
    u.id = "user-uuid-1"
    u.subscription_tier = None  # falls back to NEW_FREE
    return u


def _pet(species=Species.CAT, gender=Gender.FEMALE, stage=LifeStage.ADULT):
    p = MagicMock(spec=Pet)
    p.id = "pet-uuid-1"
    p.species = species
    p.gender = gender
    p.stage = stage
    p.age_in_months = None
    p.name = "Whiskers"
    return p


def _base_state(**overrides):
    state = {
        "user": _user(),
        "pet": _pet(),
        "user_message": "how often should I feed my cat?",
        "message_type": MessageType.TEXT,
        "session": {},
        "dialogue_id": "diag-uuid-1",
        "system_prompt": "You are Pawly.",
        "messages": [{"role": "user", "content": "how often should I feed my cat?"}],
        "trace_id": "trace-abc-123",
        "rule_triage": TriageLevel.GREEN,
        "matched_patterns": [],
        "response_text": "Feeding twice a day is fine.",
        "input_tokens": 10,
        "output_tokens": 20,
        "llm_triage": TriageLevel.GREEN,
        "final_triage": TriageLevel.GREEN,
        "triage_overridden": False,
        "override_direction": "",
        "intent": "nutrition",
        "sentiment": None,
        "symptom_tags": [],
    }
    state.update(overrides)
    return state


# ── rule_triage_node ──────────────────────────────────────────────────────────


async def test_rule_triage_node_green_for_normal_question() -> None:
    state = _base_state(user_message="how often should I feed my cat?")
    result = await rule_triage_node(state)
    assert result["rule_triage"] == TriageLevel.GREEN


async def test_rule_triage_node_red_for_emergency() -> None:
    state = _base_state(user_message="my cat can't breathe")
    result = await rule_triage_node(state)
    assert result["rule_triage"] == TriageLevel.RED


async def test_rule_triage_node_orange_for_vomiting() -> None:
    state = _base_state(user_message="dog has been vomiting all morning")
    result = await rule_triage_node(state)
    assert result["rule_triage"] == TriageLevel.ORANGE


# ── generate_response_node (structured output) ──────────────────────────────


async def test_generate_response_node_structured_output() -> None:
    """chat_structured returns parsed JSON fields directly."""
    mock_raw = {
        "response_text": "Twice a day is ideal.",
        "triage_level": "GREEN",
        "intent": "nutrition",
        "sentiment": "CALM",
        "symptom_tags": [],
        "input_tokens": 50,
        "output_tokens": 30,
    }
    with patch("src.llm.graph.nodes.get_gemini_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat_structured = AsyncMock(return_value=mock_raw)
        mock_get.return_value = mock_client

        result = await generate_response_node(_base_state())

    assert result["response_text"] == "Twice a day is ideal."
    assert result["llm_triage"] == TriageLevel.GREEN
    assert result["intent"] == "nutrition"
    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 30


async def test_generate_response_node_returns_symptom_tags() -> None:
    mock_raw = {
        "response_text": "This vomiting needs attention.",
        "triage_level": "ORANGE",
        "intent": "symptom_report",
        "sentiment": "ANXIOUS",
        "symptom_tags": ["vomiting"],
        "input_tokens": 40,
        "output_tokens": 25,
    }
    with patch("src.llm.graph.nodes.get_gemini_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat_structured = AsyncMock(return_value=mock_raw)
        mock_get.return_value = mock_client

        result = await generate_response_node(_base_state())

    assert result["llm_triage"] == TriageLevel.ORANGE
    assert result["symptom_tags"] == ["vomiting"]


async def test_generate_response_node_fallback_on_exception() -> None:
    with patch("src.llm.graph.nodes.get_gemini_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat_structured = AsyncMock(side_effect=Exception("API down"))
        mock_get.return_value = mock_client

        result = await generate_response_node(_base_state())

    assert "trouble connecting" in result["response_text"]
    assert result["llm_triage"] is None
    assert result["input_tokens"] == 0


# ── resolve_triage_node ───────────────────────────────────────────────────────


async def test_resolve_triage_takes_stricter_rule() -> None:
    state = _base_state(rule_triage=TriageLevel.RED, llm_triage=TriageLevel.GREEN)
    result = await resolve_triage_node(state)
    assert result["final_triage"] == TriageLevel.RED
    assert result["triage_overridden"] is True
    assert result["override_direction"] == "rules_stricter"


async def test_resolve_triage_agreement_no_override() -> None:
    state = _base_state(rule_triage=TriageLevel.ORANGE, llm_triage=TriageLevel.ORANGE)
    result = await resolve_triage_node(state)
    assert result["final_triage"] == TriageLevel.ORANGE
    assert result["triage_overridden"] is False


# ── should_override routing ───────────────────────────────────────────────────


def test_should_override_routes_to_critical_when_red_overridden() -> None:
    state = _base_state(
        final_triage=TriageLevel.RED,
        triage_overridden=True,
        override_direction="rules_stricter",
    )
    assert should_override(state) == "critical_override"


def test_should_override_routes_to_finalize_when_not_overridden() -> None:
    state = _base_state(
        final_triage=TriageLevel.RED,
        triage_overridden=False,
        override_direction="",
    )
    assert should_override(state) == "finalize"


def test_should_override_routes_to_finalize_for_orange() -> None:
    state = _base_state(
        final_triage=TriageLevel.ORANGE,
        triage_overridden=True,
        override_direction="rules_stricter",
    )
    assert should_override(state) == "finalize"


def test_should_override_routes_to_finalize_for_llm_stricter() -> None:
    """llm_stricter RED doesn't trigger override — LLM was already urgent."""
    state = _base_state(
        final_triage=TriageLevel.RED,
        triage_overridden=True,
        override_direction="llm_stricter",
    )
    assert should_override(state) == "finalize"


# ── Full graph integration (mocked LLM + memory) ──────────────────────────────


async def test_full_graph_green_path() -> None:
    """End-to-end graph run with mocked Gemini and empty memory context."""
    from src.llm.graph.graph import build_graph

    mock_raw = {
        "response_text": "Twice a day is fine.",
        "triage_level": "GREEN",
        "intent": "nutrition",
        "sentiment": "CALM",
        "symptom_tags": [],
        "input_tokens": 40,
        "output_tokens": 25,
    }

    with (
        patch("src.llm.graph.nodes.load_pet_context", new_callable=AsyncMock, return_value={}),
        patch("src.llm.graph.nodes.load_related_memories", new_callable=AsyncMock, return_value=[]),
        patch("src.llm.graph.nodes.get_gemini_client") as mock_get,
        patch("src.llm.graph.nodes._store_triage_record", new_callable=AsyncMock),
    ):
        mock_client = MagicMock()
        mock_client.chat_structured = AsyncMock(return_value=mock_raw)
        mock_get.return_value = mock_client

        graph = build_graph()
        final = await graph.ainvoke(
            {
                "user": _user(),
                "pet": _pet(),
                "user_message": "how often should I feed my cat?",
                "message_type": MessageType.TEXT,
                "session": {},
                "dialogue_id": "diag-1",
            }
        )

    assert final["response_text"] == "Twice a day is fine."
    assert final["final_triage"] == TriageLevel.GREEN
    assert final["trace_id"] != ""  # UUID was generated


async def test_full_graph_red_path_triggers_override() -> None:
    """RED + rules_stricter override → critical_override node re-calls Gemini."""
    from src.llm.graph.graph import build_graph

    # First call: LLM says GREEN (benign response)
    normal_resp = {
        "response_text": "Looks fine, monitor at home.",
        "triage_level": "GREEN",
        "intent": "symptom_report",
        "sentiment": "CALM",
        "symptom_tags": [],
        "input_tokens": 30,
        "output_tokens": 20,
    }
    # Second call (override): LLM says RED
    override_resp = {
        "response_text": "This is an emergency! Go to vet NOW.",
        "triage_level": "RED",
        "intent": "symptom_report",
        "sentiment": "PANIC",
        "symptom_tags": ["breathing difficulty"],
        "input_tokens": 35,
        "output_tokens": 25,
    }
    call_count = 0

    async def mock_chat_structured(**kwargs):
        nonlocal call_count
        call_count += 1
        return override_resp if call_count > 1 else normal_resp

    with (
        patch("src.llm.graph.nodes.load_pet_context", new_callable=AsyncMock, return_value={}),
        patch("src.llm.graph.nodes.load_related_memories", new_callable=AsyncMock, return_value=[]),
        patch("src.llm.graph.nodes.get_gemini_client") as mock_get,
        patch("src.llm.graph.nodes._store_triage_record", new_callable=AsyncMock),
    ):
        mock_client = MagicMock()
        mock_client.chat_structured = mock_chat_structured
        mock_get.return_value = mock_client

        graph = build_graph()
        final = await graph.ainvoke(
            {
                "user": _user(),
                "pet": _pet(),
                "user_message": "my cat can't breathe",  # RED keyword
                "message_type": MessageType.TEXT,
                "session": {},
                "dialogue_id": "diag-2",
            }
        )

    # Should have called Gemini twice (original + critical override)
    assert call_count == 2
    # Final response should have RED FLAG formatting applied
    assert "RED FLAG ALERT" in final["response_text"]
    assert final["final_triage"] == TriageLevel.RED
