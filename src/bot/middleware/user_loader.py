"""
Middleware: load or create the User + active Pet for every incoming event.

Injects into handler data:
    data["user"]        — User ORM object (created if first time)
    data["active_pet"]  — most-recently-used Pet, or None
"""

from collections.abc import Awaitable, Callable
from typing import Any
import uuid

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject
from sqlalchemy import select

from src.db.engine import get_session_factory
from src.db.models import Pet, User
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserLoaderMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        tg_user = None
        if isinstance(event, Message) and event.from_user:
            tg_user = event.from_user
        elif isinstance(event, CallbackQuery) and event.from_user:
            tg_user = event.from_user

        if tg_user is None:
            return await handler(event, data)

        telegram_id = str(tg_user.id)
        factory = get_session_factory()
        session: dict = data.get("session", {})
        preferred_pet_id = session.get("active_pet_id")

        async with factory() as db:
            created = False
            # ── Load or create User ─────────────────────────────────────────
            result = await db.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()

            if user is None:
                created = True
                user = User(
                    telegram_id=telegram_id,
                    telegram_username=tg_user.username,
                    display_name=tg_user.full_name,
                    locale=tg_user.language_code or "en",
                )
                db.add(user)
                await db.commit()
                await db.refresh(user)
                logger.info("new user created", telegram_id=telegram_id)

            # ── Load active pet ─────────────────────────────────────────────
            # 1) Use preferred pet from session when available and valid.
            active_pet = None
            if preferred_pet_id:
                try:
                    preferred_uuid = uuid.UUID(str(preferred_pet_id))
                except ValueError:
                    preferred_uuid = None
                if preferred_uuid:
                    pref_result = await db.execute(
                        select(Pet).where(
                            Pet.id == preferred_uuid,
                            Pet.user_id == user.id,
                            Pet.is_active.is_(True),
                        )
                    )
                    active_pet = pref_result.scalar_one_or_none()

            # 2) Fallback: most recently updated/created pet.
            if active_pet is None:
                pet_result = await db.execute(
                    select(Pet)
                    .where(Pet.user_id == user.id, Pet.is_active.is_(True))
                    .order_by(Pet.updated_at.desc().nulls_last(), Pet.created_at.desc())
                    .limit(1)
                )
                active_pet = pet_result.scalar_one_or_none()

        # Sync session cache with resolved IDs (session already in data from
        # SessionMiddleware which runs before this middleware).
        session["user_id"] = str(user.id)
        # Only mark as new user when freshly created; never reset to False
        # so the profile wizard is still reachable if the session expires before
        # the user completes profile setup.
        if created:
            session["is_new_user"] = True
        if active_pet:
            session["active_pet_id"] = str(active_pet.id)
        else:
            session.pop("active_pet_id", None)

        data["user"] = user
        data["active_pet"] = active_pet
        return await handler(event, data)
