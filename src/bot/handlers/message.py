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

import re
import time
import uuid
from datetime import date
from typing import Any

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select

from src.db.engine import get_session_factory
from src.db.models import (
    ChatSession,
    Dialogue,
    Message as DBMessage,
    MessageRole,
    MessageType,
    Pet,
    Species,
    Gender,
    NeuteredStatus,
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


async def load_user_pets(user_id: str) -> list[Pet]:
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(Pet)
            .where(Pet.user_id == uuid.UUID(user_id), Pet.is_active.is_(True))
        )
        return list(result.scalars().all())


def match_pets_by_name(pets: list[Pet], text: str) -> list[Pet]:
    if not text:
        return []
    lower = text.lower()
    matched: list[Pet] = []
    for pet in pets:
        name = (pet.name or "").strip()
        if not name:
            continue
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, lower):
            matched.append(pet)
    return matched


def build_pet_choice_keyboard(pets: list[Pet]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(
            text=f"{pet.name} ({pet.species.value})",
            callback_data=f"pet_select:{pet.id}",
        )]
        for pet in pets
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _parse_age_to_months(raw: str) -> int | None:
    if not raw:
        return None
    text = raw.lower().strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(year|years|yr|yrs|y)", text)
    if m:
        years = float(m.group(1))
        return int(round(years * 12))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(month|months|mo|mos|m)", text)
    if m:
        months = float(m.group(1))
        return int(round(months))
    # If no unit, assume years for small numbers, months for larger.
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        val = float(m.group(1))
        if val <= 30:
            return int(round(val * 12))
        return int(round(val))
    return None


def _parse_weight(raw: str) -> float | None:
    if not raw:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return None
    return float(m.group(1))


def parse_pet_profile(text: str) -> dict | None:
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # 1) Markdown table parsing
    for i, line in enumerate(lines):
        if "|" in line and "name" in line.lower() and "species" in line.lower():
            headers = [h.strip().lower() for h in line.strip("|").split("|")]
            row = None
            for j in range(i + 1, len(lines)):
                candidate = lines[j]
                # skip separator row like | --- | --- |
                if set(candidate.replace("|", "").replace(" ", "")) <= {"-"}:
                    continue
                if "|" in candidate:
                    row = [c.strip() for c in candidate.strip("|").split("|")]
                    break
            if row:
                data = dict(zip(headers, row))
                return data

    # 2) Key-value lines
    data: dict[str, str] = {}
    for line in lines:
        m = re.match(r"^\s*([A-Za-z _]+)\s*[:=]\s*(.+)\s*$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            data[key] = val
    return data or None


def normalize_profile_fields(raw: dict) -> dict | None:
    if not raw:
        return None
    name = raw.get("name") or raw.get("pet name")
    species_raw = raw.get("species") or raw.get("pet type") or raw.get("type")
    if not name or not species_raw:
        return None
    species_lower = str(species_raw).lower()
    if "cat" in species_lower:
        species = Species.CAT
    elif "dog" in species_lower:
        species = Species.DOG
    else:
        species = Species.OTHER

    gender_raw = str(raw.get("gender", "")).lower()
    if "male" in gender_raw:
        gender = Gender.MALE
    elif "female" in gender_raw:
        gender = Gender.FEMALE
    else:
        gender = Gender.UNKNOWN

    neutered_raw = str(raw.get("neutered", "")).lower()
    if neutered_raw in {"yes", "y", "true"} or "yes" in neutered_raw:
        neutered = NeuteredStatus.YES
    elif neutered_raw in {"no", "n", "false"} or "no" in neutered_raw:
        neutered = NeuteredStatus.NO
    else:
        neutered = NeuteredStatus.UNKNOWN

    age_months = _parse_age_to_months(str(raw.get("age", "")))
    weight = _parse_weight(str(raw.get("weight", "")))
    breed = raw.get("breed")

    return {
        "name": str(name).strip(),
        "species": species,
        "age_in_months": age_months,
        "breed": str(breed).strip() if breed else None,
        "gender": gender,
        "neutered_status": neutered,
        "weight_latest": weight,
    }


async def create_pet_profile(user_id: str, fields: dict) -> Pet:
    factory = get_session_factory()
    async with factory() as db:
        pet = Pet(
            user_id=uuid.UUID(user_id),
            name=fields["name"],
            species=fields["species"],
            age_in_months=fields.get("age_in_months"),
            breed=fields.get("breed"),
            gender=fields.get("gender") or Gender.UNKNOWN,
            neutered_status=fields.get("neutered_status") or NeuteredStatus.UNKNOWN,
            weight_latest=fields.get("weight_latest"),
            is_active=True,
        )
        db.add(pet)
        await db.commit()
        await db.refresh(pet)
    return pet


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

    # If no active pet and not in wizard, prompt to start profile creation.
    if active_pet is None and not session.get("profile_wizard_step"):
        session["awaiting_pet_profile"] = True
        session["profile_wizard_data"] = {}
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(
                    text="Create a profile for my pet",
                    callback_data="pet_profile_start",
                )]
            ]
        )
        await message.answer(
            "Before we begin, please create your pet's profile.",
            reply_markup=kb,
        )
        return

    # Profile wizard: capture text input for name or age fields.
    # Species and neutered are handled via inline buttons in callbacks.py.
    wizard_step = session.get("profile_wizard_step")
    if wizard_step in ("name", "age"):
        text = message.text.strip()
        profile = session.get("profile_wizard_data") or {}

        if wizard_step == "name":
            profile["name"] = text
        elif wizard_step == "age":
            if not _parse_age_to_months(text):
                await message.answer(
                    "Please enter age like '2 years' or '6 months'."
                )
                return
            profile["age"] = text

        session["profile_wizard_data"] = profile
        session["profile_wizard_step"] = None  # clear pending text step

        # Delete the bot's text prompt if we stored its ID
        prompt_msg_id = session.pop("profile_prompt_message_id", None)
        if prompt_msg_id:
            try:
                await message.bot.delete_message(  # type: ignore[union-attr]
                    chat_id=message.chat.id, message_id=prompt_msg_id
                )
            except Exception:
                pass

        # Edit the form message to reflect the updated value
        from src.bot.handlers.callbacks import _build_form_text, _build_form_keyboard

        form_msg_id = session.get("profile_form_message_id")
        if form_msg_id:
            try:
                await message.bot.edit_message_text(  # type: ignore[union-attr]
                    chat_id=message.chat.id,
                    message_id=form_msg_id,
                    text=_build_form_text(profile),
                    reply_markup=_build_form_keyboard(profile),
                )
            except Exception:
                # If edit fails, send a fresh form
                sent = await message.answer(
                    _build_form_text(profile),
                    reply_markup=_build_form_keyboard(profile),
                )
                session["profile_form_message_id"] = sent.message_id
        else:
            # No stored form message — send one now
            sent = await message.answer(
                _build_form_text(profile),
                reply_markup=_build_form_keyboard(profile),
            )
            session["profile_form_message_id"] = sent.message_id

        # Also delete the user's text message to keep the chat clean
        try:
            await message.delete()
        except Exception:
            pass
        return

    # Waiting for button selection — ignore free text
    if wizard_step in ("species", "neutered"):
        return

    # Resolve active pet from message text if multiple pets exist.
    pets = await load_user_pets(user_id_str)
    if len(pets) > 1:
        matched = match_pets_by_name(pets, message.text)
        if len(matched) == 1:
            active_pet = matched[0]
            pet_id_str = str(active_pet.id)
            session["active_pet_id"] = pet_id_str
        elif len(matched) > 1:
            await message.answer(
                "Which pet are we talking about?",
                reply_markup=build_pet_choice_keyboard(matched),
            )
            return
        elif not session.get("active_pet_id"):
            await message.answer(
                "Which pet are we talking about?",
                reply_markup=build_pet_choice_keyboard(pets),
            )
            return

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
