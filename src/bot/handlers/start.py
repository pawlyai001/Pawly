"""
/start command handler.

Handles deep-link parameters in the format:
    ch_{channel}_cp_{campaign}_th_{theme}_cr_{creative}_v{version}

Example Telegram link: https://t.me/PawlyBot?start=ch_ig_cp_summer_th_dog_cr_vid1_v2
"""

import re
from typing import Any

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.db.models import Pet, User
from src.llm.orchestrator import generate_opening
from src.utils.logger import get_logger

router = Router(name="start")
logger = get_logger(__name__)

# Regex to locate known prefix markers inside the start param string.
_MARKER_RE = re.compile(r"(?:^|_)(ch|cp|th|cr)_")
_VALID_PARAM_RE = re.compile(r"^[a-z0-9_]{1,64}$")
# Version suffix to strip from the end of values: _v1, _v2, _v12, …
_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def parse_start_param(param: str) -> dict[str, str] | None:
    """
    Parse a Telegram deep-link start parameter into marketing context.

    Format: ch_{channel}_cp_{campaign}_th_{theme}_cr_{creative}[_v{n}]

    Returns a dict with keys channel / campaign / theme / creative,
    or None if the param is invalid or contains no recognised keys.
    """
    if not param or not _VALID_PARAM_RE.match(param):
        return None

    key_map = {"ch": "channel", "cp": "campaign", "th": "theme", "cr": "creative"}
    result: dict[str, str] = {}

    # Strip trailing _v{n} version suffix before parsing so it doesn't bleed
    # into the last field's value.
    clean = _VERSION_SUFFIX_RE.sub("", param)

    matches = list(_MARKER_RE.finditer(clean))
    for i, m in enumerate(matches):
        short_key = m.group(1)
        val_start = m.end()
        # value runs until the start of the next marker (minus the leading _)
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
        value = clean[val_start:val_end].rstrip("_")
        if short_key in key_map and value:
            result[key_map[short_key]] = value

    return result or None


@router.message(CommandStart())
async def cmd_start(
    message: Message,
    user: User,
    active_pet: Pet | None,
    session: dict[str, Any],
) -> None:
    # ── Parse deep-link marketing context ───────────────────────────────────
    args = (message.text or "").split(maxsplit=1)
    marketing_context: dict[str, str] | None = None
    if len(args) > 1:
        marketing_context = parse_start_param(args[1])
        if marketing_context:
            session["marketing_context"] = marketing_context
            logger.info(
                "marketing context parsed",
                telegram_id=user.telegram_id,
                context=marketing_context,
            )

    is_new_user = active_pet is None

    # No pet profile yet: show the profile setup form before anything else.
    if active_pet is None:
        session["awaiting_pet_profile"] = True
        session["profile_wizard_step"] = None
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
            "Welcome to Pawly! Before we begin, let's set up your pet's profile.",
            reply_markup=kb,
        )
        return

    # ── Generate personalised opening via LLM ───────────────────────────────
    opening = await generate_opening(
        user=user,
        pet=active_pet,
        is_new_user=is_new_user,
        marketing_context=marketing_context,
    )

    await message.answer(opening.response_text, parse_mode=None)
    logger.info(
        "start handled",
        telegram_id=user.telegram_id,
        is_new_user=is_new_user,
        has_marketing=marketing_context is not None,
    )
