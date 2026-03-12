"""
Admin-only bot commands.
"""

import pathlib

import yaml
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.config import settings
from src.llm.prompts.system import reload_prompt_sections
from src.utils.logger import get_logger

router = Router(name="admin")
logger = get_logger(__name__)
_PROMPT_FILE = pathlib.Path(__file__).parents[2] / "llm" / "prompts" / "prompts_config.yaml"
_SECTION_ORDER = ("identity", "conversation_rules", "hard_rules", "medical_format")


def _is_admin(telegram_id: str) -> bool:
    raw = settings.admin_telegram_ids.strip()
    if not raw:
        return False
    allow = {item.strip() for item in raw.split(",") if item.strip()}
    return telegram_id in allow


def _load_prompt_sections() -> dict:
    with _PROMPT_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {key: str(data.get(key, "")).rstrip("\n") for key in _SECTION_ORDER}


def _write_prompt_sections(sections: dict) -> None:
    lines: list[str] = []
    for key in _SECTION_ORDER:
        value = str(sections.get(key, "")).rstrip("\n")
        lines.append(f"{key}: |")
        if value:
            for line in value.splitlines():
                lines.append(f"  {line}")
        lines.append("")
    _PROMPT_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@router.message(Command("reload_prompt"))
async def cmd_reload_prompt(message: Message) -> None:
    user_id = str(message.from_user.id) if message.from_user else ""
    if not _is_admin(user_id):
        await message.answer("Not authorized for /reload_prompt.", parse_mode=None)
        return

    sections = reload_prompt_sections()
    logger.info("prompt reloaded", user_id=user_id)
    await message.answer(
        "Prompt reloaded. Sections: "
        f"{', '.join(sections.keys())}.",
        parse_mode=None,
    )


@router.message(Command("prompt_sections"))
async def cmd_prompt_sections(message: Message) -> None:
    user_id = str(message.from_user.id) if message.from_user else ""
    if not _is_admin(user_id):
        await message.answer("Not authorized for /prompt_sections.", parse_mode=None)
        return

    await message.answer(
        "Editable sections: identity, conversation_rules, hard_rules, medical_format",
        parse_mode=None,
    )


@router.message(Command("prompt_show"))
async def cmd_prompt_show(message: Message) -> None:
    user_id = str(message.from_user.id) if message.from_user else ""
    if not _is_admin(user_id):
        await message.answer("Not authorized for /prompt_show.", parse_mode=None)
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or parts[1].strip() not in _SECTION_ORDER:
        await message.answer(
            "Usage: /prompt_show <section>\n"
            "Sections: identity, conversation_rules, hard_rules, medical_format",
            parse_mode=None,
        )
        return

    section = parts[1].strip()
    sections = _load_prompt_sections()
    await message.answer(sections.get(section, ""), parse_mode=None)


@router.message(Command("prompt_set"))
async def cmd_prompt_set(message: Message) -> None:
    user_id = str(message.from_user.id) if message.from_user else ""
    if not _is_admin(user_id):
        await message.answer("Not authorized for /prompt_set.", parse_mode=None)
        return

    raw = message.text or ""
    header, *body_lines = raw.splitlines()
    header_parts = header.split(maxsplit=1)
    if len(header_parts) < 2:
        await message.answer(
            "Usage:\n/prompt_set <section>\n<new content>",
            parse_mode=None,
        )
        return

    section = header_parts[1].strip()
    if section not in _SECTION_ORDER:
        await message.answer(
            "Unknown section. Use /prompt_sections to list valid sections.",
            parse_mode=None,
        )
        return

    new_content = "\n".join(body_lines).rstrip()
    if not new_content:
        await message.answer(
            "No content provided. Put the new prompt text on the lines after the command.",
            parse_mode=None,
        )
        return

    sections = _load_prompt_sections()
    sections[section] = new_content
    _write_prompt_sections(sections)
    reload_prompt_sections()
    await message.answer(f"Updated and reloaded: {section}", parse_mode=None)
