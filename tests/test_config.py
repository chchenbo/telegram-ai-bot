import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ai_bot.config import (  # noqa: E402
    DEFAULT_MODEL,
    Config,
    ConfigError,
    load_env_file,
    parse_user_ids,
)

BASE_ENV = {"TELEGRAM_BOT_TOKEN": "123456:ABC-def", "DEEPSEEK_API_KEY": "sk-test"}


class ParseUserIdsTest(unittest.TestCase):
    def test_multiple_separators(self):
        self.assertEqual(parse_user_ids("1, 2;3\n4"), frozenset({1, 2, 3, 4}))

    def test_blank(self):
        self.assertEqual(parse_user_ids("   "), frozenset())

    def test_rejects_garbage(self):
        with self.assertRaises(ConfigError):
            parse_user_ids("12,abc")


class ConfigTest(unittest.TestCase):
    def test_minimal_env(self):
        with patch.dict(os.environ, BASE_ENV, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.default_model, DEFAULT_MODEL)
        self.assertFalse(cfg.default_thinking)
        self.assertTrue(cfg.is_setup_mode)
        self.assertEqual(cfg.history_max_messages, 40)

    def test_missing_token(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk"}, clear=True), self.assertRaises(
            ConfigError
        ) as ctx:
            Config.from_env()
        self.assertIn("TELEGRAM_BOT_TOKEN", str(ctx.exception))

    def test_missing_api_key(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "1:a"}, clear=True), self.assertRaises(
            ConfigError
        ) as ctx:
            Config.from_env()
        self.assertIn("API_KEY", str(ctx.exception))

    def test_malformed_token_is_rejected(self):
        env = dict(BASE_ENV, TELEGRAM_BOT_TOKEN="not-a-token")
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError):
            Config.from_env()

    def test_whitelist_and_switches(self):
        env = dict(
            BASE_ENV,
            ALLOWED_USER_IDS="111,222",
            DEFAULT_THINKING="true",
            DEFAULT_MODEL="deepseek-v4-pro",
            MAX_OUTPUT_TOKENS="4096",
        )
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.allowed_user_ids, frozenset({111, 222}))
        self.assertTrue(cfg.default_thinking)
        self.assertEqual(cfg.default_model, "deepseek-v4-pro")
        self.assertEqual(cfg.max_output_tokens, 4096)
        self.assertFalse(cfg.is_setup_mode)

    def test_bad_number_is_reported_clearly(self):
        env = dict(BASE_ENV, HISTORY_MAX_MESSAGES="lots")
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError) as ctx:
            Config.from_env()
        self.assertIn("HISTORY_MAX_MESSAGES", str(ctx.exception))

    def test_edit_interval_floor(self):
        env = dict(BASE_ENV, EDIT_INTERVAL="0.1")
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError):
            Config.from_env()

    def test_token_is_masked(self):
        with patch.dict(os.environ, BASE_ENV, clear=True):
            cfg = Config.from_env()
        self.assertNotIn("ABC-def", cfg.masked_token())


class EnvFileTest(unittest.TestCase):
    def test_reads_values_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / ".env"
            path.write_text(
                "# 注释\n"
                "export FOO=bar\n"
                'QUOTED="hello world"\n'
                "EMPTY=\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["FOO"], "bar")
                self.assertEqual(os.environ["QUOTED"], "hello world")

    def test_does_not_override_real_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / ".env"
            path.write_text("FOO=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {"FOO": "from-shell"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["FOO"], "from-shell")



class ProviderConfigTest(unittest.TestCase):
    def test_defaults_to_deepseek(self):
        with patch.dict(os.environ, BASE_ENV, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.provider, "deepseek")
        self.assertTrue(cfg.supports_thinking)
        self.assertTrue(cfg.send_stream_options)

    def test_openai_requires_an_explicit_model(self):
        env = {"TELEGRAM_BOT_TOKEN": "1:a", "API_KEY": "sk", "PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError):
            Config.from_env()

    def test_openai_with_model(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "1:a",
            "API_KEY": "sk",
            "PROVIDER": "openai",
            "DEFAULT_MODEL": "gpt-4.1",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.provider, "openai")
        self.assertFalse(cfg.supports_thinking)
        self.assertTrue(cfg.send_stream_options)

    def test_anthropic_provider(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "1:a",
            "API_KEY": "sk-ant",
            "PROVIDER": "anthropic",
            "DEFAULT_MODEL": "claude-sonnet-4-5",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.provider, "anthropic")
        self.assertTrue(cfg.supports_thinking)
        self.assertFalse(cfg.send_stream_options)
        self.assertEqual(cfg.api_base_url, "https://api.anthropic.com")

    def test_base_url_defaults_follow_the_provider(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "1:a",
            "API_KEY": "sk",
            "PROVIDER": "openai",
            "DEFAULT_MODEL": "gpt-4.1",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.api_base_url, "https://api.openai.com/v1")

    def test_unknown_provider_is_rejected(self):
        env = {"TELEGRAM_BOT_TOKEN": "1:a", "API_KEY": "sk", "PROVIDER": "nonsense", "DEFAULT_MODEL": "x"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigError) as ctx:
            Config.from_env()
        self.assertIn("PROVIDER", str(ctx.exception))

    def test_api_key_alias_wins(self):
        env = {
            "TELEGRAM_BOT_TOKEN": "1:a",
            "API_KEY": "sk-new",
            "DEEPSEEK_API_KEY": "sk-old",
            "DEFAULT_MODEL": "deepseek-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.api_key, "sk-new")


if __name__ == "__main__":
    unittest.main()
