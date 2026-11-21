import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
from ai_bot.llm import (  # noqa: E402
    AnthropicClient,
    ChatClient,
    ChatError,
    Delta,
    Usage,
    build_messages,
    iter_sse_payloads,
    parse_chunk,
)
from ai_bot.telegram import TelegramClient, TelegramError  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_first_then_history_then_user(self):
        messages = build_messages(
            "You are an assistant",
            [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ],
            "this question",
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "this question")

    def test_blank_system_prompt_is_omitted(self):
        messages = build_messages("", [], "hi")
        self.assertEqual(messages, [{"role": "user", "content": "hi"}])

    def test_image_content_blocks_are_kept_verbatim(self):
        blocks = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        messages = build_messages("s", [], blocks)
        self.assertEqual(messages[-1]["content"], blocks)


class SseParsingTest(unittest.TestCase):
    def test_ignores_comments_and_blank_lines(self):
        lines = [": ping", "", "data: {\"a\":1}", "event: x", "data: [DONE]", "data: {\"b\":2}"]
        self.assertEqual(list(iter_sse_payloads(lines)), ['{"a":1}'])

    def test_handles_crlf_and_spaces(self):
        lines = ["data:  {\"a\": 1}  \r"]
        self.assertEqual(list(iter_sse_payloads(lines)), ["{\"a\": 1}"])

    def test_parse_content_and_reasoning(self):
        payload = json.dumps(
            {"choices": [{"delta": {"reasoning_content": "thinking", "content": "answer"}}]}
        )
        deltas, usage = parse_chunk(payload)
        self.assertEqual(
            [(d.kind, d.text) for d in deltas],
            [("reasoning", "thinking"), ("content", "answer")],
        )
        self.assertIsNone(usage)

    def test_parse_usage(self):
        payload = json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            }
        )
        _, usage = parse_chunk(payload)
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens), (12, 5, 8))

    def test_parse_malformed_chunk_is_ignored(self):
        deltas, usage = parse_chunk("{oops")
        self.assertEqual(deltas, [])
        self.assertIsNone(usage)


class StreamTest(unittest.IsolatedAsyncioTestCase):
    def _client(self, handler):
        return ChatClient(
            "sk-test",
            "https://api.deepseek.com",
            transport=httpx.MockTransport(handler),
        )

    async def test_stream_yields_deltas_then_usage(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            body = (
                'data: {"choices":[{"delta":{"reasoning_content":"first thought"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":", world"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":10,"completion_tokens":3}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

        client = self._client(handler)
        try:
            events = []
            async for event in client.stream(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                thinking=False,
            ):
                events.append(event)
        finally:
            await client.aclose()

        deltas = [e for e in events if isinstance(e, Delta)]
        usages = [e for e in events if isinstance(e, Usage)]
        self.assertEqual("".join(d.text for d in deltas if d.kind == "content"), "Hello, world")
        self.assertEqual([d.text for d in deltas if d.kind == "reasoning"], ["first thought"])
        self.assertEqual(usages[0].completion_tokens, 3)

        self.assertEqual(seen["model"], "deepseek-flash")
        self.assertEqual(seen["thinking"], {"type": "disabled"})
        self.assertNotIn("temperature", seen)
        self.assertTrue(seen["stream"])
        self.assertEqual(seen["stream_options"], {"include_usage": True})

    async def test_thinking_mode_payload(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=b"data: [DONE]\n\n")

        client = self._client(handler)
        try:
            async for _ in client.stream(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": "hi"}],
                thinking=True,
                max_tokens=2048,
            ):
                pass
        finally:
            await client.aclose()

        self.assertEqual(seen["thinking"], {"type": "enabled"})
        self.assertEqual(seen["reasoning_effort"], "high")
        self.assertEqual(seen["max_tokens"], 2048)

    async def test_insufficient_balance_is_explained(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json={"error": {"message": "Insufficient Balance", "type": "insufficient_balance"}},
            )

        client = self._client(handler)
        with self.assertRaises(ChatError) as ctx:
            async for _ in client.stream(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                thinking=False,
            ):
                pass
        await client.aclose()
        self.assertEqual(ctx.exception.status, 402)
        self.assertIn("credit", str(ctx.exception))

    async def test_connection_error_is_wrapped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        client = self._client(handler)
        with self.assertRaises(ChatError) as ctx:
            async for _ in client.stream(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                thinking=False,
            ):
                pass
        await client.aclose()
        self.assertIn("reach", str(ctx.exception))


class TelegramClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_not_modified_is_not_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: message is not modified",
                },
            )

        client = TelegramClient("123:abc", transport=httpx.MockTransport(handler))
        try:
            await client.edit_message_text(1, 2, "same text")
        finally:
            await client.aclose()

    async def test_other_errors_raise(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"ok": False, "error_code": 403, "description": "Forbidden: bot was blocked"},
            )

        client = TelegramClient("123:abc", transport=httpx.MockTransport(handler))
        with self.assertRaises(TelegramError):
            await client.send_message(1, "hi")
        await client.aclose()

    async def test_send_message_returns_message_id(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 77}})

        client = TelegramClient("123:abc", transport=httpx.MockTransport(handler))
        try:
            self.assertEqual(await client.send_message(1, "hi"), 77)
        finally:
            await client.aclose()



class ProviderPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_style_sends_thinking(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=b"data: [DONE]\n\n")

        client = ChatClient("sk", "https://api.deepseek.com", transport=httpx.MockTransport(handler))
        async for _ in client.stream(model="deepseek-flash", messages=[{"role": "user", "content": "hi"}]):
            pass
        await client.aclose()

        self.assertEqual(seen["thinking"], {"type": "disabled"})
        self.assertEqual(seen["stream_options"], {"include_usage": True})

    async def test_generic_style_omits_proprietary_fields(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=b"data: [DONE]\n\n")

        client = ChatClient(
            "sk",
            "https://api.example.com/v1",
            transport=httpx.MockTransport(handler),
            send_thinking=False,
            send_stream_options=False,
        )
        async for _ in client.stream(model="llama3", messages=[{"role": "user", "content": "hi"}]):
            pass
        await client.aclose()

        self.assertNotIn("thinking", seen)
        self.assertNotIn("reasoning_effort", seen)
        self.assertNotIn("stream_options", seen)

    def test_openai_style_reasoning_key_is_understood(self):
        payload = json.dumps({"choices": [{"delta": {"reasoning": "hmm"}}]})
        deltas, _ = parse_chunk(payload)
        self.assertEqual([(d.kind, d.text) for d in deltas], [("reasoning", "hmm")])



class AnthropicClientTest(unittest.IsolatedAsyncioTestCase):
    def _client(self, handler):
        return AnthropicClient(
            "sk-ant",
            "https://api.anthropic.com",
            transport=httpx.MockTransport(handler),
        )

    async def test_converts_messages_and_parses_events(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            seen["headers"] = dict(request.headers)
            body = (
                'event: message_start\n'
                'data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}\n\n'
                'event: content_block_delta\n'
                'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}\n\n'
                'event: content_block_delta\n'
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n'
                'event: message_delta\n'
                'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
                'event: message_stop\n'
                'data: {"type":"message_stop"}\n\n'
            )
            return httpx.Response(200, content=body.encode())

        client = self._client(handler)
        events = []
        async for event in client.stream(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
            ],
        ):
            events.append(event)
        await client.aclose()

        deltas = [e for e in events if isinstance(e, Delta)]
        usages = [e for e in events if isinstance(e, Usage)]
        self.assertEqual([d.text for d in deltas if d.kind == "content"], ["Hello"])
        self.assertEqual([d.text for d in deltas if d.kind == "reasoning"], ["hmm"])
        self.assertEqual(usages[0].prompt_tokens, 12)
        self.assertEqual(usages[0].completion_tokens, 7)

        self.assertEqual(seen["system"], "be nice")
        self.assertEqual(seen["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(seen["max_tokens"], 4096)
        self.assertTrue(seen["stream"])
        self.assertEqual(seen["headers"]["x-api-key"], "sk-ant")
        self.assertEqual(seen["headers"]["anthropic-version"], "2023-06-01")

    async def test_image_url_becomes_a_base64_block(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=b'data: {"type":"message_stop"}\n\n')

        client = self._client(handler)
        async for _ in client.stream(
            model="claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ],
        ):
            pass
        await client.aclose()

        blocks = seen["messages"][0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "what is this"})
        self.assertEqual(
            blocks[1],
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        )

    async def test_api_error_is_explained(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
                },
            )

        client = self._client(handler)
        with self.assertRaises(ChatError) as ctx:
            async for _ in client.stream(
                model="claude-sonnet-4-5", messages=[{"role": "user", "content": "hi"}]
            ):
                pass
        await client.aclose()
        self.assertIn("invalid x-api-key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
