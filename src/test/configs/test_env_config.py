import pytest

from src.configs.env_config import (
    BaseConfig,
    DevConfig,
    EnvTestConfig,
    GlobalConfig,
    ProdConfig,
    get_config,
)


def test_base_config(monkeypatch):
    """Test BaseConfig with test environment"""
    config = BaseConfig()
    assert hasattr(config, "ENV_STATE")
    assert config.ENV_STATE == "test"


def test_global_config_defaults():
    config = GlobalConfig()
    assert config.OPENAI_API_KEY is None
    assert config.ALLOWED_HOSTS == ""
    assert config.REDIS_URL is None
    assert config.VALKEY_URL is None


def test_global_config_allowed_hosts():
    config = GlobalConfig(ALLOWED_HOSTS="localhost,example.com")
    assert config.get_allowed_hosts == ["localhost", "example.com"]

    config = GlobalConfig(ALLOWED_HOSTS="")
    assert config.get_allowed_hosts == []


def test_environment_specific_configs():
    dev_config = DevConfig()
    prod_config = ProdConfig()
    test_config = EnvTestConfig()

    assert dev_config.model_config["env_prefix"] == "DEV_"
    assert prod_config.model_config["env_prefix"] == "PROD_"
    assert test_config.model_config["env_prefix"] == "TEST_"


@pytest.mark.parametrize(
    "env_state,expected_config",
    [
        ("dev", DevConfig),
        ("prod", ProdConfig),
        ("test", EnvTestConfig),
    ],
)
def test_get_config(env_state, expected_config):
    config = get_config(env_state)
    assert isinstance(config, expected_config)


def test_get_config_invalid_env():
    with pytest.raises(KeyError):
        get_config("invalid")


@pytest.mark.parametrize(
    "env_vars,expected_values",
    [
        (
            {
                "TEST_OPENAI_API_KEY": "test-key",
                "TEST_ALLOWED_HOSTS": "localhost,test.com",
                "TEST_REDIS_URL": "redis://localhost",
                "TEST_VALKEY_URL": "http://valkey",
            },
            {
                "OPENAI_API_KEY": "test-key",
                "ALLOWED_HOSTS": "localhost,test.com",
                "REDIS_URL": "redis://localhost",
                "VALKEY_URL": "http://valkey",
            },
        ),
    ],
)
def test_environment_variables(env_vars, expected_values, monkeypatch):
    # Clear the lru_cache for get_config
    get_config.cache_clear()

    # Set test environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Create fresh config instance
    config = EnvTestConfig()
    for key, value in expected_values.items():
        assert getattr(config, key) == value
