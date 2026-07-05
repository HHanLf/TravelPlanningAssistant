from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Travel Planning Assistant"
    app_env: str = "development"
    app_debug: bool = True
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    llm_provider: str = "mock"
    llm_model: str = "travel-planner-v1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1200

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_model: str = "qwen-plus"
    qwen_audio_model: str = "qwen-audio-turbo"
    qwen_embedding_model: str = "text-embedding-v4"

    asr_provider: str = "aliyun_nls"
    aliyun_nls_app_key: str = "..."
    aliyun_nls_token: str = ""
    aliyun_nls_url: str = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
    aliyun_access_key_id: str = ""
    aliyun_access_key_secret: str = ""
    aliyun_nls_region: str = "cn-shanghai"

    redis_url: str = "redis://localhost:6379/0"
    use_redis_memory: bool = False
    conversation_history_limit: int = 20
    conversation_memory_ttl_seconds: int = 86400
    user_profile_ttl_seconds: int = 604800

    default_city: str = "杭州"
    default_trip_days: int = 3

    tencent_map_api_key: str = ""
    ctrip_api_key: str = ""
    search_api_key: str = ""
    weather_api_key: str = ""
    justoneapi_token: str = ""
    justoneapi_base_url: str = "https://api.justoneapi.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
