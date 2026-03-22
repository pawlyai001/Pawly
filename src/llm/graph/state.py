"""
PawlyState — the shared state dict that flows through the LangGraph pipeline.

All fields are optional (total=False) so each node only needs to declare
the fields it reads/writes. The initial invocation provides the required
input fields; nodes progressively fill in the rest.
"""

from typing import Any, Optional
from typing_extensions import TypedDict

from src.db.models import MessageType, Pet, RiskLevel, Sentiment, TriageLevel, User


class PawlyState(TypedDict, total=False):
    # ── Required inputs (provided at graph entry) ─────────────────────────────
    user: User
    pet: Optional[Pet]
    user_message: str
    message_type: MessageType
    session: dict[str, Any]
    dialogue_id: str

    # ── Set by load_context node ───────────────────────────────────────────────
    pet_context: dict[str, Any]      # raw context bundle from reader.py
    system_prompt: str               # built system prompt string
    messages: list[dict[str, Any]]   # recent_turns + current user message
    trace_id: str                    # UUID for tracing

    # ── Set by rule_triage node (runs in parallel with load_context) ──────────
    rule_triage: TriageLevel
    matched_patterns: list[str]

    # ── Set by generate_response node (structured output from LLM) ────────────
    response_text: str
    llm_triage: Optional[TriageLevel]   # from structured JSON output
    intent: Optional[str]
    symptom_tags: list[str]
    sentiment: Optional[Sentiment]
    input_tokens: int
    output_tokens: int

    # ── Set by resolve_triage node ─────────────────────────────────────────────
    final_triage: TriageLevel
    triage_overridden: bool
    override_direction: str          # "" | "rules_stricter" | "llm_stricter"

    # ── Set by finalize node ──────────────────────────────────────────────────
    risk_level: Optional[RiskLevel]
