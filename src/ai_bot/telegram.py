"""A minimal async wrapper around the Telegram Bot API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
FILE_BASE = "https://api.telegram.org/file"


class TelegramError(RuntimeError):
    def __init__(self, method: str, description: str, code: int | None = None) -> None:
        super().__init__(f"{method}: {description}")
        self.method = method
        self.description = description
        self.code = code


class TelegramRateLimited(TelegramError):
    def __init__(self, method: str, description: str, retry_after: float) -> None:
        super().__init__(method, description, 429)
        self.retry_after = retry_after


class TelegramClient:
    def __init__(
        self,
        token: str,
        api_base: str = API_BASE,
        file_base: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._file_base = (file_base or f"{self._api_base}/file").rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0, read=120.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def _bot_url(self) -> str:
        return f"{self._api_base}/bot{self._token}"

    async def call(
        self,
        method: str,
        *,
        request_timeout: float | None = None,
        retries: int = 2,
        **params: Any,
    ) -> Any:
        url = f"{self._bot_url}/{method}"
        attempt = 0
        while True:
            try:
                kwargs: dict[str, Any] = {"json": params}
                if request_timeout is not None:
                    kwargs["timeout"] = request_timeout
                response = await self._client.post(url, **kwargs)
            except httpx.HTTPError as exc:
                raise TelegramError(method, f"network error: {exc}") from exc

            try:
                data = response.json()
            except ValueError as exc:
                raise TelegramError(
                    method,
                    f"HTTP {response.status_code}：{response.text[:200]}",
                    response.status_code,
                ) from exc

            if data.get("ok"):
                return data.get("result")

            code = data.get("error_code", response.status_code)
            description = str(data.get("description") or "unknown error")
            if code == 429 and attempt < retries:
                retry_after = float((data.get("parameters") or {}).get("retry_after", 3))
                attempt += 1
                log.warning("Telegram rate limit on %s, retrying in %.1fs", method, retry_after)
                await asyncio.sleep(retry_after + 0.5)
                continue
            if code == 429:
                retry_after = float((data.get("parameters") or {}).get("retry_after", 3))
                raise TelegramRateLimited(method, description, retry_after)
            raise TelegramError(method, description, code)

    # ---- API methods ----

    async def get_me(self) -> dict:
        return await self.call("getMe")

    async def get_updates(
        self,
        *,
        offset: int | None = None,
        timeout: int = 30,
        limit: int = 100,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "timeout": timeout,
            "limit": limit,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            params["offset"] = offset
        result = await self.call(
            "getUpdates",
            request_timeout=timeout + 20,
            **params,
        )
        return result or []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
    ) -> int:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        if reply_to is not None:
            params["reply_parameters"] = {
                "message_id": reply_to,
                "allow_sending_without_reply": True,
            }
        result = await self.call("sendMessage", **params)
        return int(result["message_id"])

    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        try:
            await self.call(
                "editMessageText",
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                link_preview_options={"is_disabled": True},
            )
        except TelegramError as exc:
            # Telegram answers 400 when the content is unchanged; that is not an error.
            if "not modified" in exc.description:
                return
            raise

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        try:
            await self.call("sendChatAction", retries=0, chat_id=chat_id, action=action)
        except TelegramError as exc:
            log.debug("Could not send the typing action: %s", exc)

    async def set_my_commands(self, commands: list[dict]) -> None:
        await self.call("setMyCommands", commands=commands)

    async def get_file(self, file_id: str) -> dict:
        return await self.call("getFile", file_id=file_id)

    async def download(self, file_path: str) -> bytes:
        url = f"{self._file_base}/bot{self._token}/{file_path}"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TelegramError("download", f"download failed: {exc}") from exc
        return response.content
