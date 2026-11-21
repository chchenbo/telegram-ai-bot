"""Configuration: environment variables first, then .env in the working directory."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "deepseek-flash"
PROVIDERS = ("deepseek", "openai", "anthropic")
DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant reachable through Telegram. "
    "Answer in the same language the user writes in, and keep it readable on a phone screen. "
    "Do not use Markdown headings, tables or code fences; plain text and short paragraphs only. "
    "Keep answers short unless the user asks for detail."
)


class ConfigError(RuntimeError):
    """Missing or malformed configuration."""


def load_env_file(path: Path) -> None:
    """Minimal .env parser: KEY=VALUE, with # comments, an optional ``export`` prefix and quotes.

    Variables already present in the environment win; the file never overrides them.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        if default is None:
            raise ConfigError(f"Missing environment variable {name}")
        return default
    return value.strip()


def _env_first(*names: str, default: str | None = None) -> str:
    """Return the first non-empty environment variable among names."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    if default is None:
        raise ConfigError(f"Missing environment variable {names[0]}")
    return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def parse_user_ids(raw: str) -> frozenset[int]:
    ids: set[int] = set()
    for part in re.split(r"[,;\s]+", raw.strip()):
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError as exc:
            raise ConfigError(f"ALLOWED_USER_IDS contains a non-numeric entry: {part!r}") from exc
    return frozenset(ids)


@dataclass(frozen=True)
class Config:
    telegram_token: str
    api_key: str
    provider: str = "deepseek"
    allowed_user_ids: frozenset[int] = field(default_factory=frozenset)
    api_base_url: str = "https://api.deepseek.com"
    telegram_api_base: str = "https://api.telegram.org"
    default_model: str = DEFAULT_MODEL
    default_thinking: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    state_path: Path = Path("data/state.json")
    history_max_messages: int = 40
    history_max_chars: int = 48000
    max_output_tokens: int | None = None
    edit_interval: float = 1.2
    poll_timeout: int = 30
    request_timeout: float = 300.0

    @classmethod
    def from_env(cls) -> Config:
        token = _env("TELEGRAM_BOT_TOKEN")
        provider = _env("PROVIDER", "deepseek").lower()
        if provider not in PROVIDERS:
            raise ConfigError(f"PROVIDER must be one of {', '.join(PROVIDERS)}, got {provider!r}")
        key = _env_first("API_KEY", "DEEPSEEK_API_KEY")
        base_url = _env_first("API_BASE_URL", "DEEPSEEK_BASE_URL", default=DEFAULT_BASE_URLS[provider])
        if ":" not in token:
            raise ConfigError(
                "TELEGRAM_BOT_TOKEN does not look like a bot token (expected 123456:ABC-DEF...)"
            )

        max_tokens_raw = os.environ.get("MAX_OUTPUT_TOKENS", "").strip()
        max_tokens = None
        if max_tokens_raw:
            try:
                max_tokens = int(max_tokens_raw)
            except ValueError as exc:
                raise ConfigError(f"MAX_OUTPUT_TOKENS must be an integer, got {max_tokens_raw!r}") from exc

        model = os.environ.get("DEFAULT_MODEL", "").strip()
        if not model:
            if provider != "deepseek":
                raise ConfigError("DEFAULT_MODEL must be set when PROVIDER is not deepseek")
            model = DEFAULT_MODEL
        cfg = cls(
            telegram_token=token,
            api_key=key,
            provider=provider,
            allowed_user_ids=parse_user_ids(os.environ.get("ALLOWED_USER_IDS", "")),
            api_base_url=base_url.rstrip("/"),
            telegram_api_base=_env("TELEGRAM_API_BASE", "https://api.telegram.org").rstrip("/"),
            default_model=model,
            default_thinking=_env_bool("DEFAULT_THINKING", False),
            system_prompt=os.environ.get("SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT,
            state_path=Path(_env("STATE_PATH", "data/state.json")),
            history_max_messages=_env_int("HISTORY_MAX_MESSAGES", 40),
            history_max_chars=_env_int("HISTORY_MAX_CHARS", 48000),
            max_output_tokens=max_tokens,
            edit_interval=_env_float("EDIT_INTERVAL", 1.2),
            poll_timeout=_env_int("POLL_TIMEOUT", 30),
            request_timeout=_env_float("REQUEST_TIMEOUT", 300.0),
        )
        if cfg.edit_interval < 0.5:
            raise ConfigError(
                "EDIT_INTERVAL must be at least 0.5, otherwise Telegram will rate-limit the edits"
            )
        if cfg.poll_timeout < 5:
            raise ConfigError("POLL_TIMEOUT must be at least 5")
        return cfg

    @property
    def supports_thinking(self) -> bool:
        """DeepSeek and Anthropic both expose a reasoning mode; OpenAI's is automatic."""
        return self.provider in {"deepseek", "anthropic"}

    @property
    def send_stream_options(self) -> bool:
        """include_usage is an OpenAI-ism; Anthropic reports usage in its own events."""
        return self.provider in {"deepseek", "openai"}

    @property
    def is_setup_mode(self) -> bool:
        """With an empty allowlist only /whoami works, so nobody can spend your credits."""
        return not self.allowed_user_ids

    def masked_token(self) -> str:
        """Show the numeric bot id (public) and mask the secret part entirely."""
        head, _, _tail = self.telegram_token.partition(":")
        return f"{head}:{'*' * 10}"
