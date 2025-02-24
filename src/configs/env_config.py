import pathlib
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent


class BaseConfig(BaseSettings):
    ENV_STATE: Optional[str] = None
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        secrets_dir=str(ROOT_DIR),
    )


class GlobalConfig(BaseConfig):
    OPENAI_API_KEY: Optional[str] = None
    ALLOWED_HOSTS: str = ""
    REDIS_URL: Optional[str] = None
    VALKEY_URL: Optional[str] = None

    @property
    def get_allowed_hosts(self) -> list[str]:
        return (
            [host.strip() for host in self.ALLOWED_HOSTS.split(",")]
            if self.ALLOWED_HOSTS
            else []
        )


class DevConfig(GlobalConfig):
    ENV_STATE: str = "dev"
    model_config = SettingsConfigDict(env_prefix="DEV_")


class ProdConfig(GlobalConfig):
    ENV_STATE: str = "prod"
    model_config = SettingsConfigDict(env_prefix="PROD_")


class EnvTestConfig(GlobalConfig):
    ENV_STATE: str = "test"
    model_config = SettingsConfigDict(env_prefix="TEST_")


@lru_cache()
def get_config():
    base = BaseConfig()
    env_state = base.ENV_STATE or "prod"
    configs = {"dev": DevConfig, "prod": ProdConfig, "test": EnvTestConfig}
    return configs.get(env_state, ProdConfig)()


# Use the updated configuration
config = get_config()
