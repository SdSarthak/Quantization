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
        # uvicorn raises on an unknown level, so catch it at config time.
        {"AIRAVATA_LOG_LEVEL": "verbose"},
        # 70000 binds nowhere; the socket error would come much later.
        {"AIRAVATA_PORT": "70000"},
    ],
)
def test_invalid_values_raise_config_error(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_validate_runs_again_after_fields_are_reassigned():
    """Assigning to a dataclass field bypasses ``__post_init__``."""
    settings = Settings()
    settings.port = 99999
    with pytest.raises(ConfigError, match="port"):
        settings.validate()

    settings.port = 8000
    settings.log_level = "chatty"
    with pytest.raises(ConfigError, match="log_level"):
        settings.validate()


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


def test_parse_dotenv_handles_comments_quotes_and_export():
    from airavata_quant.config import parse_dotenv

    parsed = parse_dotenv(
        "\n".join(
            [
                "# a comment",
                "",
                "AIRAVATA_PORT=9100",
                'export AIRAVATA_MODEL_NAME="org/quoted"',
                "AIRAVATA_HOST = '0.0.0.0' ",
                "not a pair",
                "=novalue",
            ]
        )
    )
    assert parsed == {
        "AIRAVATA_PORT": "9100",
        "AIRAVATA_MODEL_NAME": "org/quoted",
        "AIRAVATA_HOST": "0.0.0.0",
    }


def test_load_dotenv_returns_empty_for_a_missing_file(tmp_path):
    from airavata_quant.config import load_dotenv

    assert load_dotenv(tmp_path / "absent.env") == {}


def test_load_dotenv_reads_a_real_file(tmp_path):
    from airavata_quant.config import load_dotenv

    path = tmp_path / ".env"
    path.write_text("AIRAVATA_DEVICE=cpu\n", encoding="utf-8")
    assert load_dotenv(path) == {"AIRAVATA_DEVICE": "cpu"}


def test_process_environment_overrides_the_dotenv_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "AIRAVATA_PORT=1111\nAIRAVATA_MODEL_NAME=org/from-file\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIRAVATA_PORT", "2222")
    settings = Settings.from_env()
    assert settings.port == 2222
    assert settings.model_name == "org/from-file"
