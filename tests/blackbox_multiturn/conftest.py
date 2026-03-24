import asyncio
import json
import os
import re
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiogram import BaseMiddleware, Bot, Dispatcher
from aiogram.types import Chat, Message, TelegramObject, Update, User as TgUser
from aiogram.fsm.storage.memory import MemoryStorage

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-bot-token")
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from src.config import settings
from src.bot.handlers import message as message_handler
from src.bot.middleware.rate_limiter import RateLimiterMiddleware
from src.bot.middleware.session import SessionMiddleware
from src.db.models import (
    Gender,
    MemorySource,
    MemoryTerm,
    MemoryType,
    NeuteredStatus,
    Pet,
    PetMemory,
    Species,
    SubscriptionTier,
    User,
)
from src.llm.client import GeminiClient, get_gemini_client


TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _species(value: str) -> Species:
    return Species(value)


def _gender(value: str) -> Gender:
    return Gender(value)


def _neutered_status(value: str) -> NeuteredStatus:
    return NeuteredStatus(value)


def _memory_type(value: str) -> MemoryType:
    return MemoryType(value)


def _memory_term(value: str) -> MemoryTerm:
    return MemoryTerm(value)


def _build_pet_memory(pet_id: uuid.UUID, item: dict[str, Any]) -> PetMemory:
    return PetMemory(
        id=uuid.uuid4(),
        pet_id=pet_id,
        memory_type=_memory_type(item["memory_type"]),
        memory_term=_memory_term(item["memory_term"]),
        field=item["field"],
        value=item["value"],
        confidence_score=item.get("confidence_score", 0.9),
        source=MemorySource.AI_EXTRACTED,
        source_message_id=None,
        is_active=True,
    )


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    candidates: list[str] = [stripped]

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            body = "\n".join(lines[1:]).strip()
            if body:
                candidates.append(body)
                candidates.append(body.removesuffix("```").strip())

    def _first_balanced_json_chunk(value: str, opener: str, closer: str) -> str | None:
        start = value.find(opener)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(value)):
            char = value[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return value[start : index + 1].strip()
        return None

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1].strip())
        balanced = _first_balanced_json_chunk(stripped, opener, closer)
        if balanced:
            candidates.append(balanced)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    return stripped


class GeminiDeepEvalModelMixin:
    def __init__(self, client: GeminiClient, model_name: str | None = None) -> None:
        self._client = client
        super().__init__(model=model_name)

    def load_model(self) -> "GeminiDeepEvalModelMixin":
        return self

    def _schema_prompt(self, prompt: str) -> str:
        return (
            f"{prompt}\n\n"
            "Return ONLY valid JSON that matches the requested schema. "
            "Do not use markdown code fences. "
            "Do not add any commentary before or after the JSON."
        )

    def _validate_schema_response(self, schema: Any, text: str) -> Any:
        payload = _extract_json_payload(text)
        try:
            return schema.model_validate_json(payload)
        except Exception as exc:
            raise RuntimeError(
                f"DeepEval judge response could not be parsed as JSON. "
                f"Extracted payload: {payload!r}. Raw response: {text!r}"
            ) from exc

    def _generate_with_schema_retry(self, prompt: str, schema: Any) -> Any:
        attempts = [
            prompt,
            self._schema_prompt(prompt),
            self._schema_prompt(
                f"{prompt}\n\nYour previous answer was invalid or truncated. "
                "Retry and output the full JSON object only."
            ),
        ]
        last_error: Exception | None = None
        for attempt_prompt in attempts:
            result = asyncio.run(
                self._client.chat(
                    system_prompt="",
                    messages=[{"role": "user", "content": attempt_prompt}],
                    temperature=0,
                    max_tokens=4096,
                )
            )
            text = result["text"]
            try:
                return self._validate_schema_response(schema, text)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is None:
            result = asyncio.run(
                self._client.chat(
                    system_prompt="",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            )
            return result["text"]
        return self._generate_with_schema_retry(prompt, schema)

    async def _a_generate_with_schema_retry(self, prompt: str, schema: Any) -> Any:
        attempts = [
            prompt,
            self._schema_prompt(prompt),
            self._schema_prompt(
                f"{prompt}\n\nYour previous answer was invalid or truncated. "
                "Retry and output the full JSON object only."
            ),
        ]
        last_error: Exception | None = None
        for attempt_prompt in attempts:
            result = await self._client.chat(
                system_prompt="",
                messages=[{"role": "user", "content": attempt_prompt}],
                temperature=0,
                max_tokens=4096,
            )
            text = result["text"]
            try:
                return self._validate_schema_response(schema, text)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is None:
            result = await self._client.chat(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return result["text"]
        return await self._a_generate_with_schema_retry(prompt, schema)

    def get_model_name(self) -> str:
        return "gemini-deepeval-judge"


class FakeBot:
    def __init__(self) -> None:
        self.chat_actions: list[dict[str, Any]] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.deleted_messages: list[dict[str, Any]] = []
        self.edited_messages: list[dict[str, Any]] = []


class InMemoryPipeline:
    def __init__(self, redis: "InMemoryRedis") -> None:
        self.redis = redis
        self.ops: list[tuple[str, tuple[Any, ...]]] = []

    def incr(self, key: str) -> "InMemoryPipeline":
        self.ops.append(("incr", (key,)))
        return self

    def expire(self, key: str, ttl: int) -> "InMemoryPipeline":
        self.ops.append(("expire", (key, ttl)))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for op, args in self.ops:
            if op == "incr":
                results.append(await self.redis.incr(*args))
            elif op == "expire":
                results.append(await self.redis.expire(*args))
        self.ops.clear()
        return results


class InMemoryRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value
        return True

    async def incr(self, key: str) -> int:
        current = int(self.store.get(key, "0"))
        current += 1
        self.store[key] = str(current)
        return current

    async def expire(self, key: str, ttl: int) -> bool:
        return True

    def pipeline(self) -> InMemoryPipeline:
        return InMemoryPipeline(self)


class TestUserContextMiddleware(BaseMiddleware):
    def __init__(self, user: User, pet: Pet) -> None:
        self.user = user
        self.pet = pet

    async def __call__(self, handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        session = data.setdefault("session", {})
        session["user_id"] = str(self.user.id)
        session["active_pet_id"] = str(self.pet.id)
        data["user"] = self.user
        data["active_pet"] = self.pet
        return await handler(event, data)


class ConversationRuntime:
    def __init__(self, pet: Pet, memories: list[PetMemory], recent_turns: list[dict[str, str]]) -> None:
        self.pet = pet
        self.memories = memories
        self.recent_turns = list(recent_turns)

    def record_exchange(self, user_text: str, assistant_text: str) -> None:
        self.recent_turns.append({"role": "user", "content": user_text})
        self.recent_turns.append({"role": "assistant", "content": assistant_text})


@pytest.fixture(scope="session")
def real_gemini_client() -> GeminiClient:
    if not settings.google_api_key.strip():
        pytest.skip("GOOGLE_API_KEY is required for DeepEval black-box tests.")
    return get_gemini_client()


@pytest.fixture(scope="session")
def deepeval_model(real_gemini_client: GeminiClient) -> Any:
    deepeval = pytest.importorskip("deepeval.models")
    base_cls = getattr(deepeval, "DeepEvalBaseLLM")
    model_cls = type("GeminiDeepEvalModel", (GeminiDeepEvalModelMixin, base_cls), {})
    return model_cls(real_gemini_client)


@pytest.fixture(scope="session")
def load_test_cases() -> Callable[[str], list[dict[str, Any]]]:
    def _loader(filename: str) -> list[dict[str, Any]]:
        path = TEST_DATA_DIR / filename
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return _loader


@pytest.fixture
def build_user_and_pet() -> Callable[[dict[str, Any]], tuple[User, Pet]]:
    def _builder(case: dict[str, Any]) -> tuple[User, Pet]:
        user_id = uuid.uuid4()
        pet_id = uuid.uuid4()
        pet_profile = case["pet_profile"]

        user = User(
            id=user_id,
            telegram_id=f"eval-{user_id.hex[:10]}",
            display_name=case.get("user_display_name", "Eval User"),
            subscription_tier=SubscriptionTier.PLUS,
        )
        pet = Pet(
            id=pet_id,
            user_id=user_id,
            name=pet_profile["name"],
            species=_species(pet_profile["species"]),
            breed=pet_profile.get("breed"),
            age_in_months=pet_profile.get("age_in_months"),
            gender=_gender(pet_profile.get("gender", "unknown")),
            neutered_status=_neutered_status(pet_profile.get("neutered_status", "unknown")),
            weight_latest=pet_profile.get("weight_latest"),
        )
        return user, pet

    return _builder


@pytest.fixture
def build_update() -> Callable[[str, int, int], Update]:
    def _builder(text: str, message_id: int, telegram_user_id: int) -> Update:
        return Update(
            update_id=message_id,
            message=Message(
                message_id=message_id,
                date=datetime.now(),
                chat=Chat(id=telegram_user_id, type="private"),
                from_user=TgUser(
                    id=telegram_user_id,
                    is_bot=False,
                    first_name="Margaret",
                    username="margaret_eval",
                    language_code="en",
                ),
                text=text,
            ),
        )

    return _builder


@pytest.fixture
def mock_multiturn_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, Any], User, Pet], ConversationRuntime]:
    from src.llm import orchestrator

    def _apply(case: dict[str, Any], user: User, pet: Pet) -> ConversationRuntime:
        memories = [_build_pet_memory(pet.id, item) for item in case.get("memories", [])]
        runtime = ConversationRuntime(
            pet=pet,
            memories=memories,
            recent_turns=case.get("recent_turns", []),
        )
        fake_session_id = uuid.uuid4()
        fake_dialogue_id = uuid.uuid4()

        async def _fake_load_pet_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "pet": pet,
                "long_term_memories": [m for m in runtime.memories if m.memory_term == MemoryTerm.LONG],
                "mid_term_memories": [m for m in runtime.memories if m.memory_term == MemoryTerm.MID],
                "short_term_memories": [m for m in runtime.memories if m.memory_term == MemoryTerm.SHORT],
                "recent_turns": runtime.recent_turns,
                "daily_summary": None,
                "weekly_summary": None,
                "pending_confirmations": [],
            }

        async def _fake_load_related_memories(*args: Any, **kwargs: Any) -> list[PetMemory]:
            return [m for m in runtime.memories if m.memory_term == MemoryTerm.SHORT]

        async def _fake_store_triage_record(*args: Any, **kwargs: Any) -> None:
            return None

        async def _fake_load_user_pets(user_id: str) -> list[Pet]:
            return [pet]

        async def _fake_store_raw_message(*args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(id=uuid.uuid4())

        async def _fake_get_or_create_session(user_id: str) -> Any:
            return SimpleNamespace(id=fake_session_id)

        async def _fake_get_or_create_dialogue(session_id: str, pet_id: str | None) -> Any:
            return SimpleNamespace(id=fake_dialogue_id)

        async def _fake_store_enriched_messages(*args: Any, **kwargs: Any) -> None:
            return None

        async def _fake_enqueue_extraction(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(orchestrator, "load_pet_context", _fake_load_pet_context)
        monkeypatch.setattr(orchestrator, "load_related_memories", _fake_load_related_memories)
        monkeypatch.setattr(orchestrator, "_store_triage_record", _fake_store_triage_record)
        monkeypatch.setattr(message_handler, "load_user_pets", _fake_load_user_pets)
        monkeypatch.setattr(message_handler, "store_raw_message", _fake_store_raw_message)
        monkeypatch.setattr(message_handler, "get_or_create_session", _fake_get_or_create_session)
        monkeypatch.setattr(message_handler, "get_or_create_dialogue", _fake_get_or_create_dialogue)
        monkeypatch.setattr(message_handler, "store_enriched_messages", _fake_store_enriched_messages)
        monkeypatch.setattr(message_handler, "enqueue_extraction", _fake_enqueue_extraction)
        return runtime

    return _apply


@pytest.fixture
def build_router_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[User, Pet], tuple[Bot, Dispatcher, FakeBot, InMemoryRedis]]:
    async def _fake_bot_call(self: Bot, method: Any, request_timeout: int | None = None) -> Any:
        api: FakeBot = getattr(self, "_fake_api")
        method_name = method.__class__.__name__
        if method_name == "SendChatAction":
            api.chat_actions.append({"chat_id": method.chat_id, "action": method.action})
            return True
        if method_name == "SendMessage":
            payload = {
                "chat_id": method.chat_id,
                "text": method.text,
                "reply_markup": method.reply_markup,
                "parse_mode": method.parse_mode,
            }
            api.sent_messages.append(payload)
            return Message(
                message_id=len(api.sent_messages) + 1000,
                date=datetime.now(),
                chat=Chat(id=method.chat_id, type="private"),
                from_user=TgUser(id=999999, is_bot=True, first_name="Pawly", username="pawly_test_bot"),
                text=method.text,
            )
        if method_name == "DeleteMessage":
            api.deleted_messages.append({"chat_id": method.chat_id, "message_id": method.message_id})
            return True
        if method_name == "EditMessageText":
            api.edited_messages.append(
                {
                    "chat_id": method.chat_id,
                    "message_id": method.message_id,
                    "text": method.text,
                }
            )
            return True
        return True

    def _builder(user: User, pet: Pet) -> tuple[Bot, Dispatcher, FakeBot, InMemoryRedis]:
        fake_api = FakeBot()
        fake_redis = InMemoryRedis()
        bot = Bot(token="123456:TESTTOKEN")
        setattr(bot, "_fake_api", fake_api)
        monkeypatch.setattr(Bot, "__call__", _fake_bot_call)

        monkeypatch.setattr("src.db.redis.get_redis", lambda: fake_redis)
        monkeypatch.setattr("src.bot.middleware.session.get_redis", lambda: fake_redis)
        monkeypatch.setattr("src.bot.middleware.rate_limiter.get_redis", lambda: fake_redis)

        dp = Dispatcher(storage=MemoryStorage())
        dp.message.middleware(SessionMiddleware())
        dp.message.middleware(TestUserContextMiddleware(user, pet))
        dp.message.middleware(RateLimiterMiddleware())
        dp.include_router(message_handler.router)
        return bot, dp, fake_api, fake_redis

    return _builder
