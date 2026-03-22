from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    telegram_bot_token: str
    google_api_key: str
    google_maps_api_key: str = ""
    node_env: str = "development"
    port: int = 8000
    log_level: str = "info"
    main_model: str = "gemini-2.0-flash"
    extraction_model: str = "gemini-2.0-flash"
    max_turns_in_context: int = 5
    max_messages_per_minute: int = 30
    webhook_host: str = ""  # e.g. "api.pawly.app" - required in production
    telegram_proxy_url: str = ""
    admin_telegram_ids: str = ""
    prompt_hot_reload: bool = False
    use_langgraph: bool = False  # set True to enable LangGraph pipeline (experimental)

    model_config = ConfigDict(env_file=".env")


settings = Settings()

