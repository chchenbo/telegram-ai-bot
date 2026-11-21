"""Entry point: python -m ai_bot [--check]"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .bot import Bridge
from .config import Config, ConfigError, load_env_file
from .store import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-bot",
        description="Talk to DeepSeek from Telegram",
    )
    parser.add_argument("--env", default=".env", help="env file to load (default: .env)")
    parser.add_argument("--check", action="store_true", help="validate the configuration and exit")
    parser.add_argument("--log-level", default=None, help="log level, overrides LOG_LEVEL")
    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise SystemExit(f"Invalid log level: {level_name}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(Path(args.env))

    try:
        cfg = Config.from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    _configure_logging(args.log_level or os.environ.get("LOG_LEVEL", "INFO"))

    if args.check:
        allowed = ", ".join(str(i) for i in sorted(cfg.allowed_user_ids)) or "(not set — only /whoami works)"
        print(f"bot token  : {cfg.masked_token()}")
        print(f"provider   : {cfg.provider}")
        print(f"API base   : {cfg.api_base_url}")
        print(f"model      : {cfg.default_model} (thinking {'on' if cfg.default_thinking else 'off'})")
        print(f"allowlist  : {allowed}")
        print(f"state file : {cfg.state_path.resolve()}")
        return 0

    store = Store(cfg.state_path)
    store.load()
    bridge = Bridge(cfg, store)

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
