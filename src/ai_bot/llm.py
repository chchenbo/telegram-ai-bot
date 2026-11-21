"""Model clients: DeepSeek and OpenAI speak OpenAI Chat Completions, Anthropic does not.

Both implementations stream the same ``Delta`` / ``Usage`` events, so the bot layer
never has to care who is answering.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


class ChatError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Delta:
    kind: str  # "content" or "reasoning"
    text: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


_STATUS_HINTS = {
    400: "the request was rejected",
    401: "the API key is invalid or expired",
    402: "the account is out of credit",
    403: "the key is not allowed to use this model",
    404: "the endpoint or model does not exist",
    422: "malformed request",
    429: "rate limited, too many requests",
    500: "the provider had a server error",
    502: "bad gateway",
    503: "the service is temporarily unavailable",
    504: "the provider timed out",
}


def build_messages(
    system_prompt: str,
    history: Iterable[dict],
    user_content: str | list[dict],
) -> list[dict]:
    """Assemble the messages for one request. Images ride along only for this turn."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(dict(item) for item in history)
    messages.append({"role": "user", "content": user_content})
    return messages


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[str]:
    """Yield the data payloads from raw SSE lines, stopping at [DONE]."""
    for line in lines:
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield payload


def parse_chunk(payload: str) -> tuple[list[Delta], Usage | None]:
    """Parse one SSE chunk into text deltas plus any usage statistics."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        log.debug("Skipping unparsable SSE chunk: %r", payload[:200])
        return [], None

    deltas: list[Delta] = []
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        # DeepSeek calls it "reasoning_content"; some other providers use "reasoning".
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            deltas.append(Delta(kind="reasoning", text=reasoning))
        content = delta.get("content")
        if content:
            deltas.append(Delta(kind="content", text=content))

    usage = None
    raw_usage = obj.get("usage")
    if isinstance(raw_usage, dict):
        details = raw_usage.get("prompt_tokens_details") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
        )
    return deltas, usage


class ChatClient:
    """A streaming client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 300.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        send_thinking: bool = True,
        send_stream_options: bool = True,
    ) -> None:
        self._send_thinking = send_thinking
        self._send_stream_options = send_stream_options
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        thinking: bool = False,
        reasoning_effort: str = "high",
        max_tokens: int | None = None,
    ) -> AsyncIterator[Delta | Usage]:
        """Stream a completion, yielding Deltas and possibly one Usage at the end."""
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if self._send_stream_options:
            payload["stream_options"] = {"include_usage": True}
        if self._send_thinking:
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
            if thinking:
                payload["reasoning_effort"] = reasoning_effort
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ChatError(
                        self._friendly_error(response.status_code, body),
                        response.status_code,
                    )

                async for line in response.aiter_lines():
                    for payload_text in iter_sse_payloads([line]):
                        deltas, usage = parse_chunk(payload_text)
                        for delta in deltas:
                            yield delta
                        if usage is not None:
                            yield usage
        except httpx.HTTPError as exc:
            raise ChatError(f"could not reach the API: {exc}", None) from exc

    @staticmethod
    def _friendly_error(status: int, body: str) -> str:
        hint = _STATUS_HINTS.get(status, f"HTTP {status}")
        detail = ""
        try:
            parsed = json.loads(body)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
        except json.JSONDecodeError:
            detail = body.strip()
        detail = detail.strip()[:300]
        return f"{hint} ({detail})" if detail else hint


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

_ANTHROPIC_VERSION = "2023-06-01"
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.S)


def _to_anthropic_content(content: str | list[dict]) -> str | list[dict]:
    """Convert OpenAI-style content blocks into Anthropic blocks."""
    if isinstance(content, str):
        return content
    blocks: list[dict] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": block.get("text") or ""})
        elif kind == "image_url":
            url = str((block.get("image_url") or {}).get("url") or "")
            match = _DATA_URL_RE.match(url)
            if match:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": match.group("mime"),
                            "data": match.group("data"),
                        },
                    }
                )
    return blocks or [{"type": "text", "text": ""}]


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split a leading system message out and convert the rest to Anthropic's shape."""
    system_parts: list[str] = []
    converted: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            continue
        converted.append({"role": role, "content": _to_anthropic_content(message.get("content"))})
    return "\n\n".join(part for part in system_parts if part), converted


class AnthropicClient:
    """Streaming client for Anthropic's native /v1/messages API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 300.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        thinking: bool = False,
        reasoning_effort: str = "high",
        max_tokens: int | None = None,
    ) -> AsyncIterator[Delta | Usage]:
        system, converted = to_anthropic_messages(messages)

        budget = 0
        if thinking:
            max_tokens = max(max_tokens or 0, 16000)
            budget = min(8000, max_tokens // 2)

        payload: dict = {
            "model": model,
            "messages": converted,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        prompt_tokens = 0
        completion_tokens = 0
        stopped = False

        try:
            async with self._client.stream("POST", "/v1/messages", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ChatError(self._friendly_error(response.status_code, body), response.status_code)

                async for line in response.aiter_lines():
                    for payload_text in iter_sse_payloads([line]):
                        try:
                            event = json.loads(payload_text)
                        except json.JSONDecodeError:
                            continue

                        kind = event.get("type")
                        if kind == "content_block_delta":
                            delta = event.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yield Delta(kind="content", text=delta["text"])
                            elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                                yield Delta(kind="reasoning", text=delta["thinking"])
                        elif kind == "message_start":
                            usage = (event.get("message") or {}).get("usage") or {}
                            prompt_tokens = int(usage.get("input_tokens") or 0)
                        elif kind == "message_delta":
                            usage = event.get("usage") or {}
                            completion_tokens = int(usage.get("output_tokens") or 0)
                        elif kind == "message_stop":
                            stopped = True
                            break
                        elif kind == "error":
                            error = event.get("error") or {}
                            raise ChatError(
                                f"Anthropic error: {error.get('message') or error.get('type') or 'unknown'}",
                                None,
                            )
                    if stopped:
                        break
        except httpx.HTTPError as exc:
            raise ChatError(f"could not reach Anthropic: {exc}", None) from exc

        yield Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @staticmethod
    def _friendly_error(status: int, body: str) -> str:
        hint = _STATUS_HINTS.get(status, f"HTTP {status}")
        detail = ""
        try:
            parsed = json.loads(body)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("type") or "")
        except json.JSONDecodeError:
            detail = body.strip()
        detail = detail.strip()[:300]
        return f"{hint} ({detail})" if detail else hint


def build_client(cfg: Config) -> ChatClient | AnthropicClient:
    """Pick the client that matches the configured provider."""
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg.api_key, cfg.api_base_url, timeout=cfg.request_timeout)
    return ChatClient(
        cfg.api_key,
        cfg.api_base_url,
        timeout=cfg.request_timeout,
        send_thinking=cfg.supports_thinking,
        send_stream_options=cfg.send_stream_options,
    )
