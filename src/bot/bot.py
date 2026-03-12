"""
aiogram Bot + Dispatcher setup.

Middleware registration order (outer → inner):
    1. SessionMiddleware   — loads/saves Redis JSON session (outer wrapper)
    2. UserLoaderMiddleware — resolves User + active Pet from DB
    3. RateLimiterMiddleware — enforces per-user message rate limit

start_bot(bot, dp):
    development  → long polling
    production   → webhook (caller mounts the aiohttp handler on FastAPI)
"""

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.redis import RedisStorage

from src.bot.handlers import admin, callbacks, message, start
from src.bot.middleware.rate_limiter import RateLimiterMiddleware
from src.bot.middleware.session import SessionMiddleware
from src.bot.middleware.user_loader import UserLoaderMiddleware
from src.config import settings
from src.db.redis import get_redis
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def create_bot() -> tuple[Bot, Dispatcher]:
    """Create and wire up the Bot and Dispatcher."""
    session = (
        AiohttpSession(proxy=settings.telegram_proxy_url)
        if settings.telegram_proxy_url
        else None
    )
    bot = Bot(
        token=settings.telegram_bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )

    redis = get_redis()
    storage = RedisStorage(redis=redis)  # type: ignore[arg-type]
    dp = Dispatcher(storage=storage)

    # ── Middleware (registered outermost-first) ──────────────────────────────
    for observer in (dp.message, dp.callback_query):
        observer.middleware(SessionMiddleware())
        observer.middleware(UserLoaderMiddleware())

    # Rate limiter only makes sense for inbound messages, not callback queries
    dp.message.middleware(RateLimiterMiddleware())

    # ── Routers ─────────────────────────────────────────────────────────────
    dp.include_router(start.router)
    dp.include_router(message.router)
    dp.include_router(callbacks.router)
    dp.include_router(admin.router)

    me = await bot.get_me()
    logger.info("bot initialised", username=me.username, env=settings.node_env)
    return bot, dp


async def start_bot(bot: Bot, dp: Dispatcher) -> None:
    """
    Run the bot.

    Development  — long polling (blocks until cancelled).
    Production   — sets the Telegram webhook and returns immediately;
                   updates arrive via the FastAPI webhook route.
    """
    if settings.node_env != "production":
        logger.info("starting polling")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    else:
        webhook_url = f"https://{settings.webhook_host}/webhook/{settings.telegram_bot_token}"
        await bot.set_webhook(webhook_url)
        logger.info("webhook set", url=webhook_url)
        # The FastAPI route feeds updates to dp via dp.feed_webhook_update(bot, update)
