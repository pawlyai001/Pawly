"""
LangGraph node functions for the Pawly response pipeline.

Each node receives the full PawlyState dict and returns a partial dict
with only the fields it updates. No domain logic lives here — every node
delegates to the existing functions in reader.py, rules_engine.py,
client.py, prompts/, and orchestrator helpers.

Graph flow (parallel branches):
    START ──┬──→ rule_triage ──────────────────┬──→ resolve_triage ──→ [override?] ──→ finalize ──→ END
            └──→ load_context ──→ generate ────┘
"""

import uuid
from typing import Any

from src.db.models import Sentiment, SubscriptionTier, TriageLevel
from src.llm.client import get_gemini_client
from src.llm.graph.state import PawlyState
from src.llm.prompts.context import build_context_block
from src.llm.prompts.formatters import apply_response_format
from src.llm.prompts.system import build_system_prompt
from src.memory.reader import load_pet_context, load_related_memories
from src.triage.rules_engine import (
    classify_by_rules,
    compare_and_resolve,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Imported at module level so tests can patch src.llm.graph.nodes._store_triage_record
from src.llm.orchestrator import (  # noqa: E402
    _store_triage_record,
    map_triage_to_risk,
)

_HEALTH_WORDS = (
    "sick", "vomit", "diarrhea", "blood", "limp", "pain", "hurt",
    "fever", "sneeze", "cough", "lethargic", "not eating", "not drinking",
    "swollen", "wound", "scratch", "discharge", "seizure", "collapse",
    "breathing", "unconscious", "bleed",
)


def _tier(state: PawlyState) -> SubscriptionTier:
    user = state.get("user")
    return getattr(user, "subscription_tier", SubscriptionTier.NEW_FREE)


def _is_health(user_message: str) -> bool:
    lower = user_message.lower()
    return any(w in lower for w in _HEALTH_WORDS)


def _parse_triage_level(raw: str | None) -> TriageLevel | None:
    """Convert the LLM's triage string to TriageLevel enum."""
    if not raw:
        return None
    mapping = {"RED": TriageLevel.RED, "ORANGE": TriageLevel.ORANGE, "GREEN": TriageLevel.GREEN}
    return mapping.get(raw.upper())


def _parse_sentiment(raw: str | None) -> Sentiment | None:
    """Convert the LLM's sentiment string to Sentiment enum."""
    if not raw:
        return None
    mapping = {"CALM": Sentiment.CALM, "ANXIOUS": Sentiment.ANXIOUS, "PANIC": Sentiment.PANIC}
    return mapping.get(raw.upper())


# ── Node 1: load_context (runs in parallel with rule_triage) ─────────────────

async def load_context_node(state: PawlyState) -> dict[str, Any]:
    """
    Load pet memory context, build system prompt, and assemble the messages
    array. Also generates the trace_id for this turn.
    """
    trace_id = str(uuid.uuid4())
    user = state["user"]
    pet = state.get("pet")
    user_message = state["user_message"]
    session = state.get("session") or {}
    tier = _tier(state)
    pet_id_str = str(pet.id) if pet else None
    is_health = _is_health(user_message)

    ctx: dict[str, Any] = {}
    if pet:
        try:
            ctx = await load_pet_context(
                pet_id=pet_id_str,  # type: ignore[arg-type]
                user_id=str(user.id),
                tier=tier,
            )
        except Exception as exc:
            logger.warning("load_pet_context failed", error=str(exc))

        if is_health:
            try:
                related = await load_related_memories(pet_id_str, user_message)  # type: ignore[arg-type]
                existing_ids = {m.id for m in ctx.get("short_term_memories", [])}
                ctx.setdefault("short_term_memories", []).extend(
                    m for m in related if m.id not in existing_ids
                )
            except Exception as exc:
                logger.warning("load_related_memories failed", error=str(exc))

    memory_context, pending_confirmation = build_context_block(
        pet=pet,  # type: ignore[arg-type]
        long_term=ctx.get("long_term_memories", []),
        mid_term=ctx.get("mid_term_memories", []),
        short_term=ctx.get("short_term_memories", []),
        recent_turns=ctx.get("recent_turns", []),
        daily_summary=ctx.get("daily_summary"),
        pending=ctx.get("pending_confirmations", []),
    )

    system_prompt = build_system_prompt(
        user=user,
        pet=pet,
        tier=tier,
        memory_context=memory_context,
        pending_confirmation=pending_confirmation,
        marketing_context=session.get("marketing_context"),
    )

    recent_turns: list[dict] = ctx.get("recent_turns", [])
    messages = recent_turns + [{"role": "user", "content": user_message}]

    return {
        "pet_context": ctx,
        "system_prompt": system_prompt,
        "messages": messages,
        "trace_id": trace_id,
    }


# ── Node 2: rule_triage (runs in parallel with load_context) ─────────────────

async def rule_triage_node(state: PawlyState) -> dict[str, Any]:
    """Run deterministic keyword-based triage on the user message."""
    result = classify_by_rules(state.get("pet"), state["user_message"])
    return {
        "rule_triage": result.classification,
        "matched_patterns": result.matched_rules,
    }


# ── Node 3: generate_response (structured output) ───────────────────────────

async def generate_response_node(state: PawlyState) -> dict[str, Any]:
    """
    Call Gemini with structured JSON output.

    Returns response_text + LLM-classified triage, intent, sentiment,
    symptom_tags — all from a single LLM call.
    """
    client = get_gemini_client()
    try:
        raw = await client.chat_structured(
            system_prompt=state["system_prompt"],
            messages=state["messages"],
        )
        return {
            "response_text": raw.get("response_text", ""),
            "llm_triage": _parse_triage_level(raw.get("triage_level")),
            "intent": raw.get("intent"),
            "sentiment": _parse_sentiment(raw.get("sentiment")),
            "symptom_tags": raw.get("symptom_tags", []),
            "input_tokens": raw.get("input_tokens", 0),
            "output_tokens": raw.get("output_tokens", 0),
        }
    except Exception as exc:
        logger.error("llm call failed", error=str(exc))
        return {
            "response_text": (
                "I'm having trouble connecting right now. Please try again in a moment."
            ),
            "llm_triage": None,
            "intent": None,
            "sentiment": None,
            "symptom_tags": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }


# ── Node 4: resolve_triage ──────────────────────────────────────────────────

async def resolve_triage_node(state: PawlyState) -> dict[str, Any]:
    """Combine rule-based and LLM-inferred triage, taking the stricter result."""
    resolved = compare_and_resolve(
        llm_triage=state.get("llm_triage"),
        rule_classification=state.get("rule_triage", TriageLevel.GREEN),
    )
    return {
        "final_triage": resolved.final_classification,
        "triage_overridden": resolved.overridden,
        "override_direction": resolved.override_direction,
    }


def should_override(state: PawlyState) -> str:
    """Conditional edge: route to critical_override when rules escalate to RED."""
    if (
        state.get("final_triage") == TriageLevel.RED
        and state.get("triage_overridden")
        and state.get("override_direction") == "rules_stricter"
    ):
        return "critical_override"
    return "finalize"


# ── Node 5: critical_override ───────────────────────────────────────────────

async def critical_override_node(state: PawlyState) -> dict[str, Any]:
    """
    Re-call Gemini with an explicit emergency override when the rule engine
    classified RED but the LLM's structured triage was lower.

    Uses a focused override prompt to keep cost down.
    """
    override_system = (
        state["system_prompt"]
        + "\n\nCRITICAL OVERRIDE: This situation has been classified as an emergency "
        "by the safety system. You MUST treat this as URGENT. Set triage_level to RED. "
        "Push immediate vet visit. Do not suggest home treatment."
    )
    client = get_gemini_client()
    try:
        raw = await client.chat_structured(
            system_prompt=override_system,
            messages=state["messages"],
        )
        logger.warning(
            "triage RED override re-call issued",
            pet_id=str(state["pet"].id) if state.get("pet") else None,
        )
        return {
            "response_text": raw.get("response_text", state.get("response_text", "")),
            "llm_triage": TriageLevel.RED,
            "input_tokens": state.get("input_tokens", 0) + raw.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + raw.get("output_tokens", 0),
        }
    except Exception as exc:
        logger.error("RED override re-call failed", error=str(exc))
        return {}


# ── Node 6: finalize ────────────────────────────────────────────────────────

async def finalize_node(state: PawlyState) -> dict[str, Any]:
    """
    Apply visual response format and persist triage record.

    Intent, sentiment, and symptom_tags are already set by generate_response_node
    via structured output — no post-hoc keyword detection needed.
    """
    user_message = state["user_message"]
    matched = state.get("matched_patterns", [])

    final_triage = state.get("final_triage", TriageLevel.GREEN)
    rule_triage = state.get("rule_triage", TriageLevel.GREEN)
    llm_triage = state.get("llm_triage")
    risk_level = map_triage_to_risk(final_triage)

    # Merge rule-engine matched patterns into symptom_tags
    existing_tags = set(state.get("symptom_tags", []))
    for pattern in matched:
        # Strip prefix like "keyword_red:" or "symptom_orange:"
        tag = pattern.split(":", 1)[-1].replace("_", " ") if ":" in pattern else pattern
        existing_tags.add(tag)
    symptom_tags = sorted(existing_tags)

    pet = state.get("pet")
    is_health = _is_health(user_message)
    if pet and (is_health or final_triage != TriageLevel.GREEN):
        triage_dict = {
            "rule": rule_triage.value,
            "llm": llm_triage.value if llm_triage else None,
            "final": final_triage.value,
            "overridden": state.get("triage_overridden", False),
            "override_direction": state.get("override_direction", ""),
            "matched_patterns": matched,
            "confidence": 0.95 if matched else 0.5,
        }
        try:
            await _store_triage_record(
                pet_id=pet.id,
                message_id=None,
                triage=triage_dict,
                user_message=user_message,
            )
        except Exception as exc:
            logger.warning("store_triage_record failed", error=str(exc))

    # Apply Figma visual format based on resolved triage level
    formatted_response = apply_response_format(
        state.get("response_text", ""), final_triage
    )

    return {
        "response_text": formatted_response,
        "symptom_tags": symptom_tags,
        "risk_level": risk_level,
    }
