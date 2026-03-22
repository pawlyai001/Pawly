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

# Required fields — submit button is shown only when all are filled.
_REQUIRED = ("name", "species", "breed", "age", "gender", "neutered")


def _build_form_text(data: dict) -> str:
    """Render the pet profile form as a readable Telegram message."""
    name     = data.get("name") or "—"
    species  = {"cat": "🐱 Cat", "dog": "🐕 Dog"}.get(data.get("species", ""), "—")
    breed    = data.get("breed") or "—"
    age      = data.get("age") or "—"
    age_unit = data.get("age_unit", "Y")          # "Y" years | "M" months
    gender   = {"male": "Male", "female": "Female"}.get(data.get("gender", ""), "—")
    neutered = {"yes": "Yes", "no": "No", "unknown": "Not Sure"}.get(
        data.get("neutered", ""), "—"
    )
    weight      = data.get("weight") or "—"
    weight_unit = data.get("weight_unit", "kg")   # "kg" | "lb"
    med_history = data.get("medical_history") or "—"

    unit_label = "years" if age_unit == "Y" else "months"
    w_label    = weight_unit

    return (
        "🐾 Create Pet Profile\n"
        "Add your pet's basic information to receive more accurate health guidance\n\n"
        f"Pet's Name *\n{name}\n\n"
        f"Species *\n{species}\n\n"
        f"Breed *\n{breed}\n\n"
        f"Age * ({unit_label})\n{age}\n\n"
        f"Gender *\n{gender}\n\n"
        f"Neutered / Spayed *\n{neutered}\n\n"
        f"Weight (Optional, {w_label})\n{weight}\n\n"
        f"Medical History (Optional)\n{med_history}"
    )


def _build_form_keyboard(data: dict) -> InlineKeyboardMarkup:
    """Build the inline keyboard for the pet profile form."""
    name        = data.get("name")
    species     = data.get("species")
    breed       = data.get("breed")
    age         = data.get("age")
    age_unit    = data.get("age_unit", "Y")
    gender      = data.get("gender")
    neutered    = data.get("neutered")
    weight      = data.get("weight")
    weight_unit = data.get("weight_unit", "kg")

    def sp_btn(val: str, label: str) -> InlineKeyboardButton:
        tick = "✅ " if species == val else ""
        return InlineKeyboardButton(
            text=f"{tick}{label}", callback_data=f"pet_profile_species:{val}"
        )

    def gd_btn(val: str, label: str) -> InlineKeyboardButton:
        tick = "✅ " if gender == val else ""
        return InlineKeyboardButton(
            text=f"{tick}{label}", callback_data=f"pet_profile_gender:{val}"
        )

    def nt_btn(val: str, label: str) -> InlineKeyboardButton:
        tick = "✅ " if neutered == val else ""
        return InlineKeyboardButton(
            text=f"{tick}{label}", callback_data=f"pet_profile_neutered:{val}"
        )

    # Age unit toggle
    age_y_tick = "✅ " if age_unit == "Y" else ""
    age_m_tick = "✅ " if age_unit == "M" else ""

    # Weight unit toggle
    kg_tick = "✅ " if weight_unit == "kg" else ""
    lb_tick = "✅ " if weight_unit == "lb" else ""

    rows: list[list[InlineKeyboardButton]] = [
        # Pet's Name
        [
            InlineKeyboardButton(
                text=f"✏️ Name: {name}" if name else "✏️ Enter your pet's name",
                callback_data="pet_profile_set_name",
            )
        ],
        # Species — Cat / Dog
        [sp_btn("cat", "🐱 Cat"), sp_btn("dog", "🐕 Dog")],
        # Breed
        [
            InlineKeyboardButton(
                text=f"✏️ Breed: {breed}" if breed else "✏️ e.g. British Shorthair, Golden Retriever",
                callback_data="pet_profile_set_breed",
            )
        ],
        # Age + unit toggle
        [
            InlineKeyboardButton(
                text=f"🎂 Age: {age}" if age else "🎂 Enter age",
                callback_data="pet_profile_set_age",
            ),
            InlineKeyboardButton(
                text=f"{age_y_tick}Y", callback_data="pet_profile_age_unit:Y"
            ),
            InlineKeyboardButton(
                text=f"{age_m_tick}M", callback_data="pet_profile_age_unit:M"
            ),
        ],
        # Gender
        [gd_btn("male", "Male"), gd_btn("female", "Female")],
        # Neutered / Spayed
        [nt_btn("yes", "Yes"), nt_btn("no", "No"), nt_btn("unknown", "Not Sure")],
        # Weight (optional) + unit toggle
        [
            InlineKeyboardButton(
                text=f"⚖️ Weight: {weight}" if weight else "⚖️ Weight (optional)",
                callback_data="pet_profile_set_weight",
            ),
            InlineKeyboardButton(
                text=f"{kg_tick}kg", callback_data="pet_profile_weight_unit:kg"
            ),
            InlineKeyboardButton(
                text=f"{lb_tick}lb", callback_data="pet_profile_weight_unit:lb"
            ),
        ],
        # Medical History (optional)
        [
            InlineKeyboardButton(
                text="📋 Medical History: set" if data.get("medical_history")
                else "📋 Medical History (optional)",
                callback_data="pet_profile_set_medical_history",
            )
        ],
    ]

    # Submit — only once all required fields are filled
    if all(data.get(f) for f in _REQUIRED):
        rows.append(
            [
                InlineKeyboardButton(
                    text="🐾 Create Profile & Start",
                    callback_data="pet_profile_submit",
                )
            ]
        )

    return InlineKeyboardMarkup(inline_keyboard=rows)


# Alias used by step1 text builders kept for backward compat with message.py
def _build_step1_text(data: dict) -> str:
    return _build_form_text(data)


def _build_step1_keyboard(data: dict) -> InlineKeyboardMarkup:
    return _build_form_keyboard(data)


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
        pass


async def _create_pet_in_db(user_id: uuid.UUID, data: dict) -> Pet | None:
    """Persist the pet profile to the database."""
    name      = data.get("name")
    species_raw = data.get("species")
    if not name or not species_raw:
        return None

    species_map = {"cat": Species.CAT, "dog": Species.DOG}
    species = species_map.get(str(species_raw).lower(), Species.OTHER)

    gender_raw = str(data.get("gender", "unknown")).lower()
    gender = {"male": Gender.MALE, "female": Gender.FEMALE}.get(gender_raw, Gender.UNKNOWN)

    neutered_raw = str(data.get("neutered", "unknown")).lower()
    neutered = {"yes": NeuteredStatus.YES, "no": NeuteredStatus.NO}.get(
        neutered_raw, NeuteredStatus.UNKNOWN
    )

    # Parse age → months
    age_in_months: int | None = None
    age_raw  = data.get("age")
    age_unit = data.get("age_unit", "Y")
    if age_raw:
        import re

        text = str(age_raw).lower()
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            val = float(m.group(1))
            if age_unit == "M":
                age_in_months = int(round(val))
            else:
                # Try to detect unit from text; fall back to session age_unit
                if re.search(r"month|mo\b", text):
                    age_in_months = int(round(val))
                else:
                    age_in_months = int(round(val * 12))

    # Parse weight → kg (store as float)
    weight_kg: float | None = None
    weight_raw  = data.get("weight")
    weight_unit = data.get("weight_unit", "kg")
    if weight_raw:
        import re

        m = re.search(r"(\d+(?:\.\d+)?)", str(weight_raw))
        if m:
            w = float(m.group(1))
            weight_kg = round(w * 0.453592, 2) if weight_unit == "lb" else w

    factory = get_session_factory()
    async with factory() as db:
        pet = Pet(
            user_id=user_id,
            name=str(name).strip(),
            species=species,
            breed=str(data.get("breed", "")).strip() or None,
            age_in_months=age_in_months,
            gender=gender,
            neutered_status=neutered,
            weight_latest=weight_kg,
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

    # ── Pet profile: open the form ─────────────────────────────────────────────
    if data == "pet_profile_start":
        profile = session.get("profile_wizard_data") or {}
        session["profile_wizard_data"] = profile
        session["profile_wizard_step"] = None
        await callback.answer()
        if callback.message:
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

    # ── Text-input triggers ────────────────────────────────────────────────────
    _text_prompts = {
        "pet_profile_set_name":             ("name",            "What's your pet's name?"),
        "pet_profile_set_breed":            ("breed",           "What breed is your pet? (e.g. British Shorthair, Golden Retriever)"),
        "pet_profile_set_age":              ("age",             "How old is your pet? Use the Y/M buttons to switch between years and months."),
        "pet_profile_set_weight":           ("weight",          "What is your pet's weight? (Use the kg/lb buttons to select the unit)"),
        "pet_profile_set_medical_history":  ("medical_history", "Briefly describe any past illnesses, chronic conditions, or allergies (or type 'none')."),
    }
    if data in _text_prompts:
        step, prompt = _text_prompts[data]
        session["profile_wizard_step"] = step
        await callback.answer()
        if callback.message:
            sent = await callback.message.answer(prompt)
            session["profile_prompt_message_id"] = sent.message_id
        return

    # ── Toggle: age unit (Y / M) ───────────────────────────────────────────────
    if data.startswith("pet_profile_age_unit:"):
        unit = data.split(":", 1)[1]   # "Y" or "M"
        profile = session.get("profile_wizard_data") or {}
        profile["age_unit"] = unit
        session["profile_wizard_data"] = profile
        await callback.answer("Unit updated.")
        await _try_edit_form(callback, profile)
        return

    # ── Toggle: weight unit (kg / lb) ─────────────────────────────────────────
    if data.startswith("pet_profile_weight_unit:"):
        unit = data.split(":", 1)[1]   # "kg" or "lb"
        profile = session.get("profile_wizard_data") or {}
        profile["weight_unit"] = unit
        session["profile_wizard_data"] = profile
        await callback.answer("Unit updated.")
        await _try_edit_form(callback, profile)
        return

    # ── Species selection ──────────────────────────────────────────────────────
    if data.startswith("pet_profile_species:"):
        species_val = data.split(":", 1)[1]
        profile = session.get("profile_wizard_data") or {}
        profile["species"] = species_val
        session["profile_wizard_data"] = profile
        await callback.answer("Got it.")
        await _try_edit_form(callback, profile)
        return

    # ── Gender selection ───────────────────────────────────────────────────────
    if data.startswith("pet_profile_gender:"):
        gender_val = data.split(":", 1)[1]
        profile = session.get("profile_wizard_data") or {}
        profile["gender"] = gender_val
        session["profile_wizard_data"] = profile
        await callback.answer("Got it.")
        await _try_edit_form(callback, profile)
        return

    # ── Neutered / Spayed selection ────────────────────────────────────────────
    if data.startswith("pet_profile_neutered:"):
        neutered_val = data.split(":", 1)[1]
        profile = session.get("profile_wizard_data") or {}
        profile["neutered"] = neutered_val
        session["profile_wizard_data"] = profile
        await callback.answer("Got it.")
        await _try_edit_form(callback, profile)
        return

    # ── Submit / create pet ────────────────────────────────────────────────────
    if data == "pet_profile_submit":
        profile = session.get("profile_wizard_data") or {}

        # Validate all required fields before hitting the DB
        missing = [f for f in _REQUIRED if not profile.get(f)]
        if missing:
            await callback.answer(
                f"Please fill in: {', '.join(missing)}", show_alert=True
            )
            return

        pet = await _create_pet_in_db(user.id, profile)
        if not pet:
            await callback.answer("Could not create profile. Please try again.", show_alert=True)
            return

        session["active_pet_id"]       = str(pet.id)
        session["awaiting_pet_profile"] = False
        session["is_new_user"]          = False
        session["profile_wizard_step"]  = None
        session["profile_wizard_data"]  = {}
        await callback.answer("Profile created!")

        if callback.message:
            await callback.message.edit_text(
                f"✅ Profile created for {pet.name}!\n\n"
                "How can I help you today?",
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
