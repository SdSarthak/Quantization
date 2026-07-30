from pathlib import Path

import pytest

from airavata_quant.config import DEFAULT_MODEL_NAME, ConfigError, Settings


def test_defaults_when_environment_is_empty():
    settings = Settings.from_env({})
    assert settings.model_name == DEFAULT_MODEL_NAME
    assert settings.device == "auto"
    assert settings.port == 8000
    assert settings.host == "127.0.0.1"
    assert settings.preload == ["auto"]
    assert settings.hf_token is None


def test_environment_overrides_every_field():
    settings = Settings.from_env(
        {
            "AIRAVATA_MODEL_NAME": "org/tiny",
            "AIRAVATA_CACHE_DIR": "/tmp/cache",
            "AIRAVATA_DEVICE": "CPU",
            "AIRAVATA_HOST": "0.0.0.0",
            "AIRAVATA_PORT": "9001",
            "AIRAVATA_MAX_WORKERS": "2",
            "AIRAVATA_PRELOAD": "original, int4 ",
            "AIRAVATA_USE_AMP": "false",
            "AIRAVATA_LOG_LEVEL": "debug",
        }
    )
    assert settings.model_name == "org/tiny"
    assert settings.cache_dir == Path("/tmp/cache")
    assert settings.device == "cpu"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9001
    assert settings.max_workers == 2
    assert settings.preload == ["original", "int4"]
    assert settings.use_amp is False
    assert settings.log_level == "DEBUG"


def test_preload_none_disables_startup_loading():
    assert Settings.from_env({"AIRAVATA_PRELOAD": "none"}).preload == []


def test_blank_values_fall_back_to_defaults():
    settings = Settings.from_env({"AIRAVATA_MODEL_NAME": "   ", "AIRAVATA_PORT": ""})
    assert settings.model_name == DEFAULT_MODEL_NAME
    assert settings.port == 8000


@pytest.mark.parametrize(
    "env",
    [
        {"AIRAVATA_PORT": "not-a-number"},
        {"AIRAVATA_PORT": "0"},
        {"AIRAVATA_USE_AMP": "maybe"},
        {"AIRAVATA_DEVICE": "tpu"},
    ],
)
def test_invalid_values_raise_config_error(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_hf_token_read_from_prefixed_or_standard_variable():
    assert Settings.from_env({"AIRAVATA_HF_TOKEN": "hf_a"}).hf_token == "hf_a"
    assert Settings.from_env({"HF_TOKEN": "hf_b"}).hf_token == "hf_b"
    assert Settings.from_env({"HUGGING_FACE_HUB_TOKEN": "hf_c"}).hf_token == "hf_c"
    # The prefixed variable wins when both are present.
    assert (
        Settings.from_env({"AIRAVATA_HF_TOKEN": "hf_a", "HF_TOKEN": "hf_b"}).hf_token
        == "hf_a"
    )


def test_ensure_directories_creates_all_paths(tmp_path):
    settings = Settings(
        cache_dir=tmp_path / "c",
        quantized_model_path=tmp_path / "q",
        benchmark_dir=tmp_path / "b",
    )
    settings.ensure_directories()
    assert (tmp_path / "c").is_dir()
    assert (tmp_path / "q").is_dir()
    assert (tmp_path / "b").is_dir()
