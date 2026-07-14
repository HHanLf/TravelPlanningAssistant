from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Travel Planning Assistant"
    app_env: str = "development"
    app_debug: bool = True
    api_prefix: str = "/api/v1"
    cors_origins: str | list[str] = "http://localhost:5173,http://127.0.0.1:5173"

    llm_provider: str = "dashscope"
    llm_model: str = "travel-planner-v1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1200

    qwen_api_key: str = Field(default="", validation_alias=AliasChoices("QWEN_API_KEY", "DASHSCOPE_API_KEY"))
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_model: str = "qwen3.6-plus"
    qwen_audio_model: str = "qwen-audio-turbo"
    qwen_embedding_model: str = "text-embedding-v4"

    asr_provider: str = "aliyun_nls"
    aliyun_nls_app_key: str = ""
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
    reflection_max_rounds: int = 2

    tencent_map_api_key: str = ""
    tencent_map_base_url: str = "https://apis.map.qq.com"
    tencent_map_place_search_path: str = "/ws/place/v1/search"
    tencent_map_geocoder_path: str = "/ws/geocoder/v1/"
    tencent_map_direction_path_prefix: str = "/ws/direction/v1"
    tencent_map_weather_path: str = "/ws/weather/v1/"
    ctrip_api_key: str = ""
    ctrip_base_url: str = ""
    search_api_key: str = ""
    search_base_url: str = ""
    weather_api_key: str = ""
    weather_base_url: str = "https://apis.map.qq.com"

    justoneapi_token: str = ""
    justoneapi_base_url: str = "https://api.justoneapi.com"
    justoneapi_xiaohongshu_search_note_path: str = "/api/xiaohongshu/search-note/v4"

    mysql_host: str = ""
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = ""

    model_config = SettingsConfigDict(
        env_file=(
            BACKEND_ROOT / ".env.example",
            PROJECT_ROOT / ".env",
            BACKEND_ROOT / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> str | list[str]:
        if isinstance(value, list):
            return value
        if value is None:
            return ""
        return str(value)

    @property
    def cors_origin_list(self) -> list[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
