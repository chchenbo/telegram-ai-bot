"""End-to-end smoke tests: a fake Telegram API and a fake DeepSeek stream."""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
from ai_bot.bot import Bridge, should_notify  # noqa: E402
from ai_bot.config import Config  # noqa: E402
from ai_bot.llm import ChatClient  # noqa: E402
from ai_bot.store import Store  # noqa: E402
from ai_bot.telegram import TelegramClient  # noqa: E402

BASE_ENV = {
    "TELEGRAM_BOT_TOKEN": "123456:ABC-def",
    "DEEPSEEK_API_KEY": "sk-test",
    "ALLOWED_USER_IDS": "111",
    "EDIT_INTERVAL": "0.5",
}

REPLY_CHUNKS = ("Hello", " I am the fake model")


class FakeTelegram:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.sent: list[tuple[int, str]] = []
        self.payloads: list[dict] = []
        self.edits: list[tuple[int, int, str]] = []
        self.actions = 0
        self.get_me_calls = 0
        self._next_message_id = 500
        self._next_update_id = 0

    # ---- helpers ----

    def all_text(self) -> str:
        return "".join(text for _, text in self.sent) + "".join(text for _, _, text in self.edits)

    def push(
        self,
        *,
        text: str | None = None,
        chat_type: str = "private",
        chat_id: int = 555,
        user_id: int = 111,
        photo: bool = False,
        caption: str | None = None,
    ) -> None:
        self._next_update_id += 1
        message: dict = {
            "message_id": self._next_update_id,
            "date": 0,
            "chat": {"id": chat_id, "type": chat_type},
            "from": {"id": user_id, "is_bot": False, "first_name": "U"},
        }
        if text is not None:
            message["text"] = text
        if photo:
            message["photo"] = [
                {"file_id": "small", "width": 90, "height": 90},
                {"file_id": "big", "width": 1000, "height": 1000},
            ]
        if caption is not None:
            message["caption"] = caption
        self.updates.append({"update_id": self._next_update_id, "message": message})

    # ---- fake Bot API ----

    async def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rstrip("/").rsplit("/", 1)[-1]
        payload = json.loads(request.content.decode("utf-8") or "{}")

        def ok(result):
            return httpx.Response(200, json={"ok": True, "result": result})

        if method == "getMe":
            self.get_me_calls += 1
            return ok({"id": 999, "is_bot": True, "username": "testbot", "first_name": "T"})
        if method == "getUpdates":
            # the real API treats offset=-1 as "just the latest one"; here it means an empty queue
            if payload.get("offset") == -1:
                return ok([])
            if not self.updates:
                await asyncio.sleep(0.02)
                return ok([])
            return ok([self.updates.pop(0)])
        if method == "sendMessage":
            self._next_message_id += 1
            self.payloads.append(payload)
            self.sent.append((int(payload["chat_id"]), payload["text"]))
            return ok({"message_id": self._next_message_id})
        if method == "editMessageText":
            self.edits.append((int(payload["chat_id"]), int(payload["message_id"]), payload["text"]))
            return ok({"message_id": payload["message_id"]})
        if method == "sendChatAction":
            self.actions += 1
            return ok(True)
        if method == "setMyCommands":
            return ok(True)
        return ok(True)


def fake_deepseek(calls: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        body = "".join(
            f'data: {json.dumps({"choices": [{"delta": {"content": chunk}}]})}\n\n'
            for chunk in REPLY_CHUNKS
        )
        body += (
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":7,"completion_tokens":5}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    return handler


async def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


@asynccontextmanager
async def running_bridge(env: dict):
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(pathlib.Path(tmp) / "state.json")
        with patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()

        fake = FakeTelegram()
        calls: list[dict] = []
        tg = TelegramClient("123456:ABC-def", transport=httpx.MockTransport(fake.handler))
        ds = ChatClient(
            "sk-test",
            "https://api.deepseek.com",
            transport=httpx.MockTransport(fake_deepseek(calls)),
        )
        bridge = Bridge(cfg, store, telegram=tg, llm=ds, handle_signals=False)
        task = asyncio.create_task(bridge.run())
        try:
            started = await wait_for(lambda: fake.get_me_calls > 0)
            assert started, "the bridge never started"
            yield bridge, fake, calls
        finally:
            bridge._stop.set()
            await asyncio.wait_for(task, timeout=5)


class DenyThrottleTest(unittest.TestCase):
    def test_first_message_is_never_swallowed(self):
        # a freshly booted machine has a small monotonic clock
        self.assertTrue(should_notify({}, 111, 0.5))
        self.assertTrue(should_notify({}, 111, 12.0))

    def test_repeats_inside_the_window_are_dropped(self):
        self.assertFalse(should_notify({111: 10.0}, 111, 20.0))

    def test_the_window_expires(self):
        self.assertTrue(should_notify({111: 10.0}, 111, 400.0))

    def test_other_users_are_unaffected(self):
        self.assertTrue(should_notify({111: 10.0}, 222, 20.0))


class SmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_user_gets_streamed_answer(self):
        async with running_bridge(BASE_ENV) as (bridge, fake, calls):
            fake.push(text="hello")
            self.assertTrue(await wait_for(lambda: "fake model" in fake.all_text()), fake.all_text())

            state = bridge.store.get(555)
            self.assertEqual([m["role"] for m in state.history], ["user", "assistant"])
            self.assertEqual(state.history[0]["content"], "hello")
            self.assertIn("fake model", state.history[1]["content"])
            self.assertEqual(state.total_turns, 1)
            self.assertEqual(state.total_prompt_tokens, 7)
            self.assertEqual(state.total_completion_tokens, 5)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["thinking"], {"type": "disabled"})
            self.assertEqual(calls[0]["model"], "deepseek-flash")
            self.assertEqual(calls[0]["messages"][-1]["content"], "hello")

    async def test_conversation_keeps_context_between_turns(self):
        async with running_bridge(BASE_ENV) as (bridge, fake, calls):
            fake.push(text="first line")
            self.assertTrue(await wait_for(lambda: "fake model" in fake.all_text()))
            fake.push(text="second line")
            self.assertTrue(await wait_for(lambda: len(calls) == 2))

            second = calls[1]["messages"]
            contents = [m["content"] for m in second]
            self.assertIn("first line", contents)
            self.assertIn("second line", contents)
            self.assertEqual(len(bridge.store.get(555).history), 4)

    async def test_unknown_user_is_rejected_without_calling_the_api(self):
        env = dict(BASE_ENV, ALLOWED_USER_IDS="999")
        async with running_bridge(env) as (_bridge, fake, calls):
            fake.push(text="hello", user_id=111)
            self.assertTrue(await wait_for(lambda: "Not authorized" in fake.all_text()), fake.all_text())
            self.assertIn("111", fake.all_text())
            self.assertEqual(calls, [])

    async def test_setup_mode_tells_you_your_user_id(self):
        env = {k: v for k, v in BASE_ENV.items() if k != "ALLOWED_USER_IDS"}
        async with running_bridge(env) as (_bridge, fake, calls):
            fake.push(text="hello")
            expected = "ALLOWED_USER_IDS=111"
            self.assertTrue(await wait_for(lambda: expected in fake.all_text()), fake.all_text())
            self.assertEqual(calls, [])

    async def test_new_command_wipes_history(self):
        async with running_bridge(BASE_ENV) as (bridge, fake, _calls):
            fake.push(text="remember this")
            self.assertTrue(await wait_for(lambda: "fake model" in fake.all_text()))
            self.assertEqual(len(bridge.store.get(555).history), 2)

            fake.push(text="/new")
            self.assertTrue(await wait_for(lambda: "Cleared the context" in fake.all_text()), fake.all_text())
            self.assertEqual(bridge.store.get(555).history, [])

    async def test_new_command_interrupts_a_running_answer(self):
        import contextlib

        async with running_bridge(BASE_ENV) as (bridge, fake, _calls):
            bridge.store.get(555).history = [{"role": "user", "content": "old"}]

            async def forever():
                await asyncio.sleep(30)

            running = asyncio.create_task(forever())
            await asyncio.sleep(0)
            bridge._active_turns[555] = running

            fake.push(text="/new")
            self.assertTrue(await wait_for(lambda: "Stopped the reply" in fake.all_text()), fake.all_text())
            self.assertEqual(bridge.store.get(555).history, [])

            with contextlib.suppress(asyncio.CancelledError):
                await running
            self.assertTrue(running.cancelled())

    async def test_group_chatter_is_ignored_until_mentioned(self):
        async with running_bridge(BASE_ENV) as (_bridge, fake, calls):
            fake.push(text="just chatting in a group", chat_type="group")
            await asyncio.sleep(0.4)
            self.assertEqual(fake.all_text(), "")
            self.assertEqual(calls, [])

            fake.push(text="@testbot hello", chat_type="group")
            self.assertTrue(await wait_for(lambda: "fake model" in fake.all_text()), fake.all_text())
            self.assertEqual(calls[0]["messages"][-1]["content"], "hello")

    async def test_private_chat_does_not_quote_the_question(self):
        async with running_bridge(BASE_ENV) as (_bridge, fake, _calls):
            fake.push(text="hello")
            self.assertTrue(await wait_for(lambda: "fake model" in fake.all_text()), fake.all_text())
            self.assertNotIn("reply_parameters", fake.payloads[0])

    async def test_group_reply_quotes_the_question(self):
        async with running_bridge(BASE_ENV) as (_bridge, fake, _calls):
            fake.push(text="@testbot hello", chat_type="group")
            self.assertTrue(await wait_for(lambda: "fake model" in fake.all_text()), fake.all_text())
            self.assertIn("reply_parameters", fake.payloads[0])

    async def test_image_is_refused_on_model_without_vision(self):
        env = dict(BASE_ENV, DEFAULT_MODEL="deepseek-v4-pro")
        async with running_bridge(env) as (_bridge, fake, calls):
            fake.push(photo=True, caption="what is this")
            self.assertTrue(
                await wait_for(lambda: "doesn't accept images" in fake.all_text()), fake.all_text()
            )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
