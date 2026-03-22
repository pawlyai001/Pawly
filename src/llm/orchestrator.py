"""
LLM orchestration layer.

Public API consumed by bot handlers:
    generate_opening(user, pet, is_new_user, marketing_context) -> OrchestratorResult
    generate_response(user, pet, dialogue_id, user_message, ...)  -> OrchestratorResult

When USE_LANGGRAPH=true, generate_response() delegates to the LangGraph
pipeline (src/llm/graph/). Otherwise, it uses the classic sequential path.

generate_opening() always uses the classic path (no triage needed).
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import settings
from src.db.models import (
    MessageType,
    Pet,
    RiskLevel,
    Sentiment,
    SubscriptionTier,
    TriageLevel,
    TriageRecord,
    User,
)
from src.llm.client import get_gemini_client
from src.llm.prompts.context import build_context_block
from src.llm.prompts.formatters import apply_response_format
from src.llm.prompts.system import build_system_prompt
from src.memory.reader import load_pet_context, load_related_memories
from src.triage.rules_engine import (
    classify_by_rules,
    compare_and_resolve,
    detect_triage_from_response,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-built graph to avoid circular import (graph.nodes imports from this module)
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from src.llm.graph import build_graph
        _graph = build_graph()
    return _graph


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class OrchestratorResult:
    """Returned by both generate_response() and generate_opening()."""

    response_text: str
    triage_result: Optional[dict] = None       # {"rule", "llm", "final", "overridden", "matched_patterns"}
    intent: Optional[str] = None
    symptom_tags: list[str] = field(default_factory=list)
    risk_level: Optional[RiskLevel] = None
    sentiment_user: Optional[Sentiment] = None
    input_tokens: int = 0
    output_tokens: int = 0


# ── Public entry points ───────────────────────────────────────────────────────


async def generate_opening(
    user: User,
    pet: Optional[Pet],
    is_new_user: bool,
    marketing_context: Optional[dict[str, str]],
) -> OrchestratorResult:
    """
    Generate a personalised /start welcome message.

    New users get an onboarding prompt; returning users get a warm re-greeting.
    Marketing context (channel / theme) is woven in when present.
    """
    system = build_system_prompt(
        user=user,
        pet=pet,
        is_new_user=is_new_user,
        marketing_context=marketing_context,
    )

    parts: list[str] = []
    if is_new_user:
        parts.append(
            "This is the user's very first message. "
            "Give a warm, friendly welcome. "
            "Introduce yourself as Pawly, an AI pet care assistant. "
            "Ask them to tell you about their pet (name, species, age)."
        )
    else:
        name = pet.name if pet else "their pet"
        parts.append(f"Welcome the user back warmly. Mention {name} by name.")

    if marketing_context:
        ch = marketing_context.get("channel", "")
        th = marketing_context.get("theme", "")
        if ch:
            parts.append(f"The user arrived from the '{ch}' channel.")
        if th:
            parts.append(f"Their area of interest is '{th}' — subtly acknowledge this.")

    user_prompt = " ".join(parts)

    client = get_gemini_client()
    try:
        raw = await client.chat(
            system_prompt=system,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=512,
        )
        text = raw["text"]
        in_tok = raw["input_tokens"]
        out_tok = raw["output_tokens"]
    except Exception as exc:
        logger.error("generate_opening failed", error=str(exc))
        text = (
            "Hi! I'm Pawly, your AI pet care assistant. "
            "Tell me about your pet and how I can help!"
        )
        in_tok = out_tok = 0

    return OrchestratorResult(
        response_text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


async def generate_response(
    user: User,
    pet: Optional[Pet],
    dialogue_id: str,
    user_message: str,
    message_type: MessageType = MessageType.TEXT,
    session: Optional[dict[str, Any]] = None,
    raw_message_id: Optional[str] = None,
) -> OrchestratorResult:
    """
    Generate a response to a user message.

    When USE_LANGGRAPH=true, delegates to the LangGraph pipeline.
    Otherwise, uses the classic sequential orchestration path.
    """
    if settings.use_langgraph:
        return await _generate_response_graph(
            user, pet, dialogue_id, user_message, message_type, session,
        )
    return await _generate_response_classic(
        user, pet, dialogue_id, user_message, message_type, session, raw_message_id,
    )


# ── LangGraph path (USE_LANGGRAPH=true) ──────────────────────────────────────


async def _generate_response_graph(
    user: User,
    pet: Optional[Pet],
    dialogue_id: str,
    user_message: str,
    message_type: MessageType,
    session: Optional[dict[str, Any]],
) -> OrchestratorResult:
    """Delegate to the LangGraph pipeline with structured LLM output."""
    initial_state = {
        "user": user,
        "pet": pet,
        "user_message": user_message,
        "message_type": message_type,
        "session": session or {},
        "dialogue_id": dialogue_id,
    }

    try:
        final_state = await _get_graph().ainvoke(initial_state)
    except Exception as exc:
        logger.error("graph pipeline failed", error=str(exc))
        return OrchestratorResult(
            response_text="I'm having trouble connecting right now. Please try again in a moment.",
        )

    triage_result = {
        "rule": final_state.get("rule_triage", TriageLevel.GREEN).value,
        "llm": final_state["llm_triage"].value if final_state.get("llm_triage") else None,
        "final": final_state.get("final_triage", TriageLevel.GREEN).value,
        "overridden": final_state.get("triage_overridden", False),
        "override_direction": final_state.get("override_direction", ""),
        "matched_patterns": final_state.get("matched_patterns", []),
        "confidence": 0.95 if final_state.get("matched_patterns") else 0.5,
    }

    return OrchestratorResult(
        response_text=final_state.get("response_text", ""),
        triage_result=triage_result,
        intent=final_state.get("intent"),
        symptom_tags=final_state.get("symptom_tags", []),
        risk_level=final_state.get("risk_level"),
        sentiment_user=final_state.get("sentiment"),
        input_tokens=final_state.get("input_tokens", 0),
        output_tokens=final_state.get("output_tokens", 0),
    )


# ── Classic path (USE_LANGGRAPH=false, default) ──────────────────────────────


async def _generate_response_classic(
    user: User,
    pet: Optional[Pet],
    dialogue_id: str,
    user_message: str,
    message_type: MessageType = MessageType.TEXT,
    session: Optional[dict[str, Any]] = None,
    raw_message_id: Optional[str] = None,
) -> OrchestratorResult:
    """Original sequential orchestration — proven stable path."""
    tier = _tier(user)
    pet_id_str = str(pet.id) if pet else None
    is_health = looks_like_health_query(user_message)

    # ── 1. LOAD CONTEXT ───────────────────────────────────────────────────────
    ctx: dict = {}
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

    long_term = ctx.get("long_term_memories", [])
    mid_term = ctx.get("mid_term_memories", [])
    short_term = ctx.get("short_term_memories", [])
    recent_turns: list[dict] = ctx.get("recent_turns", [])
    daily_summary = ctx.get("daily_summary")
    pending = ctx.get("pending_confirmations", [])

    # ── 2. BUILD SYSTEM PROMPT ────────────────────────────────────────────────
    memory_context, pending_confirmation = build_context_block(
        pet=pet,  # type: ignore[arg-type]
        long_term=long_term,
        mid_term=mid_term,
        short_term=short_term,
        recent_turns=recent_turns,
        daily_summary=daily_summary,
        pending=pending,
    )

    system = build_system_prompt(
        user=user,
        pet=pet,
        tier=tier,
        memory_context=memory_context,
        pending_confirmation=pending_confirmation,
        marketing_context=(session or {}).get("marketing_context"),
    )

    # ── 3. BUILD MESSAGES ARRAY ───────────────────────────────────────────────
    messages = recent_turns + [{"role": "user", "content": user_message}]

    # ── 4. CALL GEMINI ────────────────────────────────────────────────────────
    client = get_gemini_client()
    try:
        raw = await client.chat(system_prompt=system, messages=messages)
        response_text = raw["text"]
        in_tok = raw["input_tokens"]
        out_tok = raw["output_tokens"]
    except Exception as exc:
        logger.error("llm call failed", error=str(exc))
        response_text = (
            "I'm having trouble connecting right now. Please try again in a moment."
        )
        in_tok = out_tok = 0

    # ── 5. POST-PROCESS TRIAGE ────────────────────────────────────────────────
    rule_result = classify_by_rules(pet, user_message)
    llm_triage = detect_triage_from_response(response_text)
    resolved = compare_and_resolve(llm_triage, rule_result.classification)

    if resolved.overridden and resolved.final_classification == TriageLevel.RED:
        override_system = (
            system
            + "\n\nCRITICAL OVERRIDE: This situation has been classified as an emergency "
            "by the safety system. You MUST treat this as URGENT. Push immediate vet visit. "
            "Do not suggest home treatment."
        )
        try:
            raw2 = await client.chat(system_prompt=override_system, messages=messages)
            response_text = raw2["text"]
            in_tok += raw2["input_tokens"]
            out_tok += raw2["output_tokens"]
            logger.warning(
                "triage RED override re-call issued",
                pet_id=pet_id_str,
                rule=rule_result.classification.value,
                llm=llm_triage.value if llm_triage else None,
            )
        except Exception as exc:
            logger.error("RED override re-call failed", error=str(exc))

    # Apply Figma visual format based on resolved triage level
    response_text = apply_response_format(response_text, resolved.final_classification)

    triage_result = {
        "rule": rule_result.classification.value,
        "llm": llm_triage.value if llm_triage else None,
        "final": resolved.final_classification.value,
        "overridden": resolved.overridden,
        "override_direction": resolved.override_direction,
        "matched_patterns": rule_result.matched_rules,
        "confidence": rule_result.confidence,
    }

    risk_level = map_triage_to_risk(resolved.final_classification)

    # Persist triage record for non-GREEN outcomes (or health queries)
    if pet and (is_health or resolved.final_classification != TriageLevel.GREEN):
        try:
            await _store_triage_record(
                pet_id=pet.id,
                message_id=raw_message_id,
                triage=triage_result,
                user_message=user_message,
            )
        except Exception as exc:
            logger.warning("store_triage_record failed", error=str(exc))

    # ── 6. DETECT INTENT + SENTIMENT + SYMPTOMS ───────────────────────────────
    intent = detect_intent(user_message)
    sentiment = detect_sentiment(user_message)
    symptom_tags = extract_symptom_keywords(user_message, rule_result.matched_patterns)

    return OrchestratorResult(
        response_text=response_text,
        triage_result=triage_result,
        intent=intent,
        symptom_tags=symptom_tags,
        risk_level=risk_level,
        sentiment_user=sentiment,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


# ── Helper functions (kept for backward compat + generate_opening) ───────────


def looks_like_health_query(text: str) -> bool:
    """Return True if the message appears to be a health / symptom concern."""
    health_words = (
        "sick", "vomit", "diarrhea", "blood", "limp", "pain", "hurt",
        "fever", "sneeze", "cough", "lethargic", "not eating", "not drinking",
        "swollen", "wound", "scratch", "discharge", "seizure", "collapse",
        "breathing", "unconscious", "bleed",
    )
    lower = text.lower()
    return any(w in lower for w in health_words)


def detect_intent(text: str) -> Optional[str]:
    """Classify user message into a coarse intent bucket."""
    lower = text.lower()
    if any(w in lower for w in ("vomit", "blood", "limp", "sick", "hurt", "pain",
                                 "sneez", "cough", "fever", "letharg", "bleed",
                                 "seizu", "collaps", "unconscious")):
        return "symptom_report"
    if any(w in lower for w in ("eat", "food", "drink", "diet", "treat", "fed",
                                 "nutrition", "kibble", "wet food")):
        return "nutrition"
    if any(w in lower for w in ("walk", "exercise", "run", "play", "active", "energy")):
        return "exercise"
    if any(w in lower for w in ("groom", "bath", "brush", "nail", "fur", "coat", "shed")):
        return "grooming"
    if any(w in lower for w in ("what", "how", "why", "when", "can", "should", "?")):
        return "question"
    return None


def detect_sentiment(text: str) -> Optional[Sentiment]:
    """Infer owner's emotional state from message text."""
    lower = text.lower()
    panic_words = (
        "emergency", "dying", "dead", "help!", "please help", "oh no",
        "scared", "terrified", "panicking", "rushed",
    )
    anxious_words = (
        "worried", "concern", "not sure", "should i", "is this normal",
        "anxious", "nervous", "stressed", "afraid",
    )
    if any(w in lower for w in panic_words):
        return Sentiment.PANIC
    if any(w in lower for w in anxious_words):
        return Sentiment.ANXIOUS
    return Sentiment.CALM


def extract_symptom_keywords(
    text: str,
    matched_patterns: list[str],
) -> list[str]:
    """
    Combine rule-engine matched patterns with simple keyword extraction.

    Returns a deduplicated list of symptom strings.
    """
    keywords: set[str] = set(matched_patterns)
    symptom_words = (
        "vomiting", "diarrhea", "lethargy", "limping", "sneezing", "coughing",
        "bleeding", "seizure", "fever", "swelling", "discharge", "pain",
        "scratching", "itching", "weight loss", "not eating", "not drinking",
    )
    lower = text.lower()
    for word in symptom_words:
        if word in lower:
            keywords.add(word)
    return sorted(keywords)


def _tier(user: User) -> SubscriptionTier:
    """Return the user's subscription tier, defaulting to NEW_FREE."""
    return getattr(user, "subscription_tier", SubscriptionTier.NEW_FREE)


def map_triage_to_risk(level: TriageLevel) -> Optional[RiskLevel]:
    """Convert resolved TriageLevel to RiskLevel for DB storage."""
    mapping = {
        TriageLevel.RED: RiskLevel.HIGH,
        TriageLevel.ORANGE: RiskLevel.MED,
        TriageLevel.GREEN: RiskLevel.LOW,
    }
    return mapping.get(level)


async def _store_triage_record(
    pet_id: uuid.UUID,
    message_id: Optional[str],
    triage: dict,
    user_message: str,
) -> None:
    """Persist a TriageRecord to the database (fire-and-forget, errors are logged)."""
    from src.db.engine import get_session_factory

    final_str = triage.get("final", "green")
    rule_str = triage.get("rule", "green")
    llm_str = triage.get("llm")

    final_level = TriageLevel(final_str) if final_str else TriageLevel.GREEN
    rule_level = TriageLevel(rule_str) if rule_str else TriageLevel.GREEN
    llm_level = TriageLevel(llm_str) if llm_str else TriageLevel.GREEN

    record = TriageRecord(
        pet_id=str(pet_id),
        message_id=message_id or str(uuid.uuid4()),
        llm_classification=llm_level,
        rule_classification=rule_level,
        final_classification=final_level,
        symptoms={
            "matched": triage.get("matched_patterns", []),
            "text_excerpt": user_message[:500],
        },
    )

    factory = get_session_factory()
    async with factory() as db:
        db.add(record)
        await db.commit()
