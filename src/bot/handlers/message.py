"""
Text message handler — orchestrates the full inbound message flow:

    1. Store raw message (append-only audit log)
    2. Get or create today's ChatSession + open Dialogue
    3. Call LLM orchestrator → OrchestratorResult
    4. Send reply, splitting on paragraph/sentence boundaries if >4000 chars
    5. Store bot reply as raw message
    6. Store enriched Message records (with triage metadata)
    7. Enqueue background extraction job (fire-and-forget)
    8. Update session counters
"""

import time
import uuid
from datetime import date
from typing import Any

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.db.engine import get_session_factory
from src.db.models import (
    ChatSession,
    Dialogue,
    Message as DBMessage,
    MessageRole,
    MessageType,
    Pet,
    RawMessage,
    User,
)
from src.jobs.pool import get_arq_pool
from src.llm.orchestrator import OrchestratorResult, generate_response
from src.utils.logger import get_logger

router = Router(name="message")
logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def split_message(text: str, max_length: int = 4000) -> list[str]:
    """
    Split *text* into chunks no longer than *max_length*.

    Priority order:
        1. Double newline  (paragraph break)
        2. Single newline
        3. ". "            (sentence break)
        4. Hard split at max_length
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    for sep in ("\n\n", "\n", ". "):
        if len(remaining) <= max_length:
            break
        parts = remaining.split(sep)
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= max_length:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        remaining = current

    # Hard-split anything still too long
    while len(remaining) > max_length:
        chunks.append(remaining[:max_length])
        remaining = remaining[max_length:]

    if remaining:
        chunks.append(remaining)

    return chunks or [text]


async def store_raw_message(
    user_id: str,
    pet_id: str | None,
    dialogue_id: str | None,
    session_id: str | None,
    role: MessageRole,
    raw_content: str,
    telegram_msg_id: str = "",
) -> RawMessage:
    factory = get_session_factory()
    async with factory() as db:
        raw = RawMessage(
            user_id=user_id,
            pet_id=pet_id,
            dialogue_id=dialogue_id or "",
            session_id=session_id or "",
            role=role,
            raw_content=raw_content,
            telegram_msg_id=telegram_msg_id or None,
        )
        db.add(raw)
        await db.commit()
        await db.refresh(raw)
    return raw


async def get_or_create_session(user_id: str) -> ChatSession:
    """Get or create the ChatSession for *user_id* (DB UUID string) and today."""
    from sqlalchemy import select

    today = date.today()
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.user_id == user_id,
                ChatSession.date == today,
            )
        )
        chat_session = result.scalar_one_or_none()
        if chat_session is None:
            chat_session = ChatSession(user_id=user_id, date=today)
            db.add(chat_session)
            await db.commit()
            await db.refresh(chat_session)
    return chat_session


async def get_or_create_dialogue(
    session_id: str,
    pet_id: str | None,
) -> Dialogue:
    """Get the open Dialogue for *session_id* or create a new one."""
    from sqlalchemy import select

    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(Dialogue).where(
                Dialogue.session_id == uuid.UUID(session_id),
                Dialogue.is_open.is_(True),
            )
        )
        dialogue = result.scalars().first()
        if dialogue is None:
            dialogue = Dialogue(
                session_id=uuid.UUID(session_id),
                pet_id=pet_id,
            )
            db.add(dialogue)
            await db.commit()
            await db.refresh(dialogue)
    return dialogue


async def store_enriched_messages(
    dialogue_id: uuid.UUID,
    user_text: str,
    result: OrchestratorResult,
) -> None:
    """Persist enriched user + bot Message records with triage metadata."""
    factory = get_session_factory()

    triage = result.triage_result or {}
    is_risk_blocked = triage.get("final") == "red"
    risk_trigger = (
        "rule_override" if triage.get("overridden")
        else ("red_triage" if is_risk_blocked else None)
    )

    async with factory() as db:
        user_msg = DBMessage(
            dialogue_id=dialogue_id,
            role=MessageRole.USER,
            content=user_text,
            message_type=MessageType.TEXT,
            intent=result.intent,
            symptom_tags=result.symptom_tags,
            risk_level=result.risk_level,
            is_risk_blocked=False,
        )
        bot_msg = DBMessage(
            dialogue_id=dialogue_id,
            role=MessageRole.BOT,
            content=result.response_text,
            message_type=MessageType.TEXT,
            is_risk_blocked=is_risk_blocked,
            risk_trigger_name=risk_trigger,
        )
        db.add(user_msg)
        db.add(bot_msg)
        await db.commit()


async def enqueue_extraction(
    user_id: str,
    pet_id: str,
    dialogue_id: str,
    message_ids: list[str],
) -> None:
    """Push a memory extraction job to the ARQ queue (returns immediately)."""
    try:
        pool = await get_arq_pool()
        await pool.enqueue_job(
            "run_extraction",
            user_id=user_id,
            pet_id=pet_id,
            dialogue_id=dialogue_id,
            message_ids=message_ids,
        )
    except Exception as exc:
        # Never let a failed enqueue crash the handler
        logger.error("failed to enqueue extraction", error=str(exc), pet_id=pet_id)


# ── Handler ───────────────────────────────────────────────────────────────────


@router.message(F.text, ~CommandStart(), ~F.text.startswith("/"))
async def handle_message(
    message: Message,
    user: User,
    active_pet: Pet | None,
    session: dict[str, Any],
) -> None:
    if not message.text:
        return

    logger.info(
        "text message received",
        telegram_id=user.telegram_id,
        length=len(message.text),
    )

    await message.bot.send_chat_action(  # type: ignore[union-attr]
        chat_id=message.chat.id, action="typing"
    )

    user_id_str = str(user.id)
    pet_id_str = str(active_pet.id) if active_pet else None

    # 1. Store raw user message (session/dialogue IDs may be empty strings on
    #    the very first message — acceptable for the append-only audit log)
    raw_msg = await store_raw_message(
        user_id=user_id_str,
        pet_id=pet_id_str,
        dialogue_id=session.get("current_dialogue_id"),
        session_id=session.get("current_session_id"),
        role=MessageRole.USER,
        raw_content=message.text,
        telegram_msg_id=str(message.message_id),
    )

    # 2. Get or create today's DB session + open dialogue
    chat_session = await get_or_create_session(user_id_str)
    dialogue = await get_or_create_dialogue(
        session_id=str(chat_session.id),
        pet_id=pet_id_str,
    )
    session["current_session_id"] = str(chat_session.id)
    session["current_dialogue_id"] = str(dialogue.id)

    # 3. Call LLM orchestrator
    result = await generate_response(
        user=user,
        pet=active_pet,
        dialogue_id=str(dialogue.id),
        user_message=message.text,
        message_type=MessageType.TEXT,
        session=session,
    )

    # 4. Send reply (split if > 4000 chars)
    for chunk in split_message(result.response_text, max_length=4000):
        await message.answer(chunk, parse_mode=None)

    # 5. Store bot reply as raw message
    bot_raw = await store_raw_message(
        user_id=user_id_str,
        pet_id=pet_id_str,
        dialogue_id=str(dialogue.id),
        session_id=str(chat_session.id),
        role=MessageRole.BOT,
        raw_content=result.response_text,
    )

    # 6. Store enriched messages with triage metadata
    await store_enriched_messages(dialogue.id, message.text, result)

    # 7. Queue extraction job — fire-and-forget (never block on the job itself)
    if active_pet:
        await enqueue_extraction(
            user_id=user_id_str,
            pet_id=pet_id_str,  # type: ignore[arg-type]
            dialogue_id=str(dialogue.id),
            message_ids=[str(raw_msg.id), str(bot_raw.id)],
        )

    # 8. Update session counters
    session["turn_count"] = session.get("turn_count", 0) + 1
    session["last_message_at"] = time.time()
