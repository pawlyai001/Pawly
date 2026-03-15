"""
Inline button callback handlers.
"""

import uuid
from typing import Any

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select

from src.db.engine import get_session_factory
from src.db.models import Gender, NeuteredStatus, Pet, Species, User
from src.utils.logger import get_logger

router = Router(name="callbacks")
logger = get_logger(__name__)


# ── Profile form helpers ───────────────────────────────────────────────────────


def _build_form_text(data: dict) -> str:
    """Render the pet profile form as a readable message."""
    name = data.get("name") or "—"
    species_val = data.get("species")
    age = data.get("age") or "—"
    neutered_val = data.get("neutered")

    species_map = {"cat": "Cat", "dog": "Dog", "other": "Other"}
    neutered_map = {"yes": "Yes", "no": "No", "unknown": "Not sure"}

    species = species_map.get(species_val, "—") if species_val else "—"
    neutered = neutered_map.get(neutered_val, "—") if neutered_val else "—"

    return (
        "Let's build a profile\n"
        "Help me understand your companion better\n\n"
        f"PET'S NAME\n{name}\n\n"
        f"SPECIES\n{species}\n\n"
        f"AGE (years)\n{age}\n\n"
        f"SPAYED / NEUTERED\n{neutered}"
    )


def _build_form_keyboard(data: dict) -> InlineKeyboardMarkup:
    """Build the inline keyboard for the pet profile form."""
    name = data.get("name")
    species = data.get("species")
    age = data.get("age")
    neutered = data.get("neutered")

    def sp_btn(val: str, label: str) -> InlineKeyboardButton:
        tick = "✅ " if species == val else ""
        return InlineKeyboardButton(
            text=f"{tick}{label}", callback_data=f"pet_profile_species:{val}"
        )

    def nt_btn(val: str, label: str) -> InlineKeyboardButton:
        tick = "✅ " if neutered == val else ""
        return InlineKeyboardButton(
            text=f"{tick}{label}", callback_data=f"pet_profile_neutered:{val}"
        )

    rows = [
        # Name field
        [
            InlineKeyboardButton(
                text=f"✏️ Name: {name}" if name else "✏️ Set Pet's Name",
                callback_data="pet_profile_set_name",
            )
        ],
        # Species picker
        [sp_btn("cat", "🐱 Cat"), sp_btn("dog", "🐶 Dog"), sp_btn("other", "Other")],
        # Age field
        [
            InlineKeyboardButton(
                text=f"🎂 Age: {age}" if age else "🎂 Set Age",
                callback_data="pet_profile_set_age",
            )
        ],
        # Neutered picker
        [
            nt_btn("yes", "Spayed/Neutered"),
            nt_btn("no", "Not Neutered"),
            nt_btn("unknown", "Not Sure"),
        ],
    ]

    # Submit button appears only once required fields are filled
    if name and species:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🐾 Create Profile & Start",
                    callback_data="pet_profile_submit",
                )
            ]
        )

    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _try_edit_form(callback: CallbackQuery, data: dict) -> None:
    """Edit the callback message in-place with the current form state."""
    if not callback.message:
        return
    try:
        await callback.message.edit_text(
            _build_form_text(data),
            reply_markup=_build_form_keyboard(data),
        )
    except TelegramBadRequest:
        # Message content unchanged or already deleted — safe to ignore
        pass


async def _create_pet_in_db(user_id: uuid.UUID, data: dict) -> Pet | None:
    """Persist the pet profile to the database."""
    name = data.get("name")
    species_raw = data.get("species")
    age_raw = data.get("age")
    if not name or not species_raw:
        return None

    species_map = {"cat": Species.CAT, "dog": Species.DOG}
    species = species_map.get(str(species_raw).lower(), Species.OTHER)

    neutered_raw = str(data.get("neutered", "unknown")).lower()
    neutered_map = {"yes": NeuteredStatus.YES, "no": NeuteredStatus.NO}
    neutered = neutered_map.get(neutered_raw, NeuteredStatus.UNKNOWN)

    # Parse age string into months (best-effort)
    age_in_months: int | None = None
    if age_raw:
        import re

        text = str(age_raw).lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|yr|y)", text)
        if m:
            age_in_months = int(round(float(m.group(1)) * 12))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:month|mo|m)", text)
            if m:
                age_in_months = int(round(float(m.group(1))))
            else:
                m = re.search(r"(\d+(?:\.\d+)?)", text)
                if m:
                    val = float(m.group(1))
                    age_in_months = int(round(val * 12 if val <= 30 else val))

    factory = get_session_factory()
    async with factory() as db:
        pet = Pet(
            user_id=user_id,
            name=str(name).strip(),
            species=species,
            age_in_months=age_in_months,
            gender=Gender.UNKNOWN,
            neutered_status=neutered,
            is_active=True,
        )
        db.add(pet)
        await db.commit()
        await db.refresh(pet)
        return pet


# ── Callback handler ───────────────────────────────────────────────────────────


@router.callback_query()
async def handle_callback(
    callback: CallbackQuery,
    user: User,
    active_pet: Pet | None,
    session: dict[str, Any],
) -> None:
    data = callback.data or ""
    logger.info("callback received", data=data, telegram_id=user.telegram_id)

    # ── Pet profile: open the form ────────────────────────────────────────────
    if data == "pet_profile_start":
        profile = session.get("profile_wizard_data") or {}
        session["profile_wizard_data"] = profile
        session["profile_wizard_step"] = None  # no pending text input
        await callback.answer()
        if callback.message:
            # Replace the "create profile" button message with the actual form
            try:
                await callback.message.edit_text(
                    _build_form_text(profile),
                    reply_markup=_build_form_keyboard(profile),
                )
                session["profile_form_message_id"] = callback.message.message_id
            except TelegramBadRequest:
                sent = await callback.message.answer(
                    _build_form_text(profile),
                    reply_markup=_build_form_keyboard(profile),
                )
                session["profile_form_message_id"] = sent.message_id
        return

    # ── Pet profile: prompt for name text ─────────────────────────────────────
    if data == "pet_profile_set_name":
        session["profile_wizard_step"] = "name"
        await callback.answer()
        if callback.message:
            sent = await callback.message.answer("What's your pet's name?")
            session["profile_prompt_message_id"] = sent.message_id
        return

    # ── Pet profile: prompt for age text ──────────────────────────────────────
    if data == "pet_profile_set_age":
        session["profile_wizard_step"] = "age"
        await callback.answer()
        if callback.message:
            sent = await callback.message.answer(
                "How old is your pet? (e.g. 2 years or 6 months)"
            )
            session["profile_prompt_message_id"] = sent.message_id
        return

    # ── Pet profile: set species ───────────────────────────────────────────────
    if data.startswith("pet_profile_species:"):
        species_val = data.split(":", 1)[1]
        profile = session.get("profile_wizard_data") or {}
        profile["species"] = species_val
        session["profile_wizard_data"] = profile
        await callback.answer("Got it.")
        await _try_edit_form(callback, profile)
        return

    # ── Pet profile: set neutered status ──────────────────────────────────────
    if data.startswith("pet_profile_neutered:"):
        neutered_val = data.split(":", 1)[1]
        profile = session.get("profile_wizard_data") or {}
        profile["neutered"] = neutered_val
        session["profile_wizard_data"] = profile
        await callback.answer("Got it.")
        await _try_edit_form(callback, profile)
        return

    # ── Pet profile: submit / create ──────────────────────────────────────────
    if data == "pet_profile_submit":
        profile = session.get("profile_wizard_data") or {}
        pet = await _create_pet_in_db(user.id, profile)
        if not pet:
            await callback.answer("Please fill in name and species first.", show_alert=True)
            return

        session["active_pet_id"] = str(pet.id)
        session["awaiting_pet_profile"] = False
        session["is_new_user"] = False
        session["profile_wizard_step"] = None
        session["profile_wizard_data"] = {}
        await callback.answer("Profile created!")

        if callback.message:
            await callback.message.edit_text(
                f"Profile created for {pet.name}! How can I help you today?",
                reply_markup=None,
            )
        return

    # ── Pet selection (multiple pets) ─────────────────────────────────────────
    if data.startswith("pet_select:"):
        pet_id = data.split(":", 1)[1]
        try:
            pet_uuid = uuid.UUID(pet_id)
        except ValueError:
            await callback.answer("Invalid pet selection.", show_alert=True)
            return

        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                select(Pet).where(
                    Pet.id == pet_uuid,
                    Pet.user_id == user.id,
                    Pet.is_active.is_(True),
                )
            )
            pet = result.scalar_one_or_none()

        if not pet:
            await callback.answer("Pet not found.", show_alert=True)
            return

        session["active_pet_id"] = str(pet.id)
        await callback.answer(f"Selected {pet.name}.")
        if callback.message:
            await callback.message.answer(
                f"Got it — we'll talk about {pet.name}. How can I help?"
            )
        return

    # Fallback
    await callback.answer(f"Action: {data}", show_alert=False)
