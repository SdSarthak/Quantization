"""Runtime configuration, read from the environment.

Every value that used to be a module level constant lives here so the service
can be pointed at a different model, cache directory or port without editing
source. All variables are optional and use the ``AIRAVATA_`` prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

ENV_PREFIX = "AIRAVATA_"
DOTENV_FILENAME = ".env"

DEFAULT_MODEL_NAME = "ai4bharat/Airavata"
VALID_DEVICES = ("auto", "cpu", "cuda")
#: Levels ``logging`` understands *and* uvicorn accepts. ``uvicorn.run`` raises
#: on anything else, so an unvalidated level turns into a traceback at startup.
VALID_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE")
MAX_PORT = 65535

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


class ConfigError(ValueError):
    """Raised when an environment variable holds an unusable value."""


def parse_dotenv(text: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Comments, blanks and ``export`` are tolerated."""
    values: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """Read a ``.env`` file if present. Missing or unreadable files yield ``{}``."""
    dotenv = Path(path or DOTENV_FILENAME)
    try:
        return parse_dotenv(dotenv.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}


def _get(env: Mapping[str, str], name: str) -> Optional[str]:
    value = env.get(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")


def _get_int(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{ENV_PREFIX}{name} must be >= {minimum}, got {value}")
    return value


@dataclass
class Settings:
    """Resolved service configuration."""

    model_name: str = DEFAULT_MODEL_NAME
    cache_dir: Path = Path("./model_cache")
    quantized_model_path: Path = Path("./quantized_models")
    benchmark_dir: Path = Path("./benchmarks")

    device: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    max_workers: int = 4
    log_level: str = "INFO"

    #: Variants to load during startup. ``["auto"]`` picks whatever the
    #: detected device supports; an empty list defers loading until the first
    #: request touches a variant.
    preload: List[str] = field(default_factory=lambda: ["auto"])
    lazy_load: bool = True
    use_amp: bool = True
    trust_remote_code: bool = True

    max_input_tokens: int = 512
    max_new_tokens: int = 2048
    max_return_sequences: int = 10

    hf_token: Optional[str] = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.quantized_model_path = Path(self.quantized_model_path)
        self.benchmark_dir = Path(self.benchmark_dir)
        self.validate()

    def validate(self) -> "Settings":
        """Check every field that something downstream would otherwise crash on.

        Called from ``__post_init__`` and again after the CLI applies its
        overrides, since assigning to a dataclass field bypasses construction.
        """
        if self.device not in VALID_DEVICES:
            raise ConfigError(
                f"device must be one of {VALID_DEVICES}, got {self.device!r}"
            )
        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}"
            )
        if not 1 <= self.port <= MAX_PORT:
            raise ConfigError(f"port must be between 1 and {MAX_PORT}, got {self.port}")
        if not self.host.strip():
            raise ConfigError("host must not be empty")
        if not self.model_name.strip():
            raise ConfigError("model_name must not be empty")
        if self.max_input_tokens < 1:
            raise ConfigError("max_input_tokens must be >= 1")
        if self.max_new_tokens < 1:
            raise ConfigError("max_new_tokens must be >= 1")
        if self.max_return_sequences < 1:
            raise ConfigError("max_return_sequences must be >= 1")
        if self.max_workers < 1:
            raise ConfigError("max_workers must be >= 1")
        return self

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from ``env`` (defaults to the process environment).

        When no mapping is supplied, a ``.env`` file in the working directory is
        layered underneath the real environment, so exported variables always
        win over the file.
        """
        if env is None:
            merged = load_dotenv()
            merged.update(os.environ)
            env = merged
        defaults = cls()

        device = (_get(env, "DEVICE") or defaults.device).lower()

        preload_raw = _get(env, "PRELOAD")
        if preload_raw is None:
            preload = list(defaults.preload)
        elif preload_raw.lower() == "none":
            preload = []
        else:
            preload = [part.strip() for part in preload_raw.split(",") if part.strip()]

        token = _get(env, "HF_TOKEN") or env.get("HF_TOKEN") or env.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

        return cls(
            model_name=_get(env, "MODEL_NAME") or defaults.model_name,
            cache_dir=Path(_get(env, "CACHE_DIR") or defaults.cache_dir),
            quantized_model_path=Path(
                _get(env, "QUANTIZED_MODEL_PATH") or defaults.quantized_model_path
            ),
            benchmark_dir=Path(_get(env, "BENCHMARK_DIR") or defaults.benchmark_dir),
            device=device,
            host=_get(env, "HOST") or defaults.host,
            port=_get_int(env, "PORT", defaults.port),
            max_workers=_get_int(env, "MAX_WORKERS", defaults.max_workers),
            log_level=(_get(env, "LOG_LEVEL") or defaults.log_level).upper(),
            preload=preload,
            lazy_load=_get_bool(env, "LAZY_LOAD", defaults.lazy_load),
            use_amp=_get_bool(env, "USE_AMP", defaults.use_amp),
            trust_remote_code=_get_bool(
                env, "TRUST_REMOTE_CODE", defaults.trust_remote_code
            ),
            max_input_tokens=_get_int(
                env, "MAX_INPUT_TOKENS", defaults.max_input_tokens
            ),
            max_new_tokens=_get_int(env, "MAX_NEW_TOKENS", defaults.max_new_tokens),
            max_return_sequences=_get_int(
                env, "MAX_RETURN_SEQUENCES", defaults.max_return_sequences
            ),
            hf_token=token.strip() if token else None,
        )

    def ensure_directories(self) -> None:
        for path in (self.cache_dir, self.quantized_model_path, self.benchmark_dir):
            path.mkdir(parents=True, exist_ok=True)
