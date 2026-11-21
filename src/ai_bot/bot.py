"""The bot itself: polling, authorization, commands, streaming rendering."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import re
import signal
import time

from .config import Config
from .llm import ChatClient, ChatError, Usage, build_client, build_messages
from .store import Store
from .telegram import TelegramClient, TelegramError, TelegramRateLimited
from .textutil import CHUNK_LIMIT, sanitize_for_telegram, split_text, take_prefix, trim_history, utf16_len

log = logging.getLogger(__name__)

COMMANDS: list[tuple[str, str]] = [
    ("new", "Forget the conversation and start over"),
    ("model", "Show or switch models: flash / pro"),
    ("think", "Toggle thinking mode"),
    ("system", "Show or set the system prompt"),
    ("stats", "Usage so far"),
    ("whoami", "Show your user id"),
    ("help", "How to use this bot"),
]

HELP_TEXT = """I'm DeepSeek, reachable from Telegram.

Talk to me like a person — I remember the conversation. You can send photos too (deepseek-flash only).

Commands:
/new — forget the context (also stops a reply that is still generating)
/model — show or switch models (any model id your provider offers)
/think — toggle thinking mode (slower and pricier, better on hard questions)
/system — show or set the system prompt
/stats — usage so far
/whoami — your user id
/help — this message

Long answers get split into several messages. The conversation is stored on the server that runs me."""

_COMMAND_RE = re.compile(r"^/([A-Za-z0-9_]+)(?:@([A-Za-z0-9_]+))?\s*(.*)$", re.S)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class StreamRenderer:
    """Render a stream into one message that keeps being edited, moving to a new one when full."""

    def __init__(
        self,
        tg: TelegramClient,
        chat_id: int,
        *,
        reply_to: int | None = None,
        min_interval: float = 1.2,
        limit: int = CHUNK_LIMIT,
    ) -> None:
        self.tg = tg
        self.chat_id = chat_id
        self.reply_to = reply_to
        self.min_interval = min_interval
        self.limit = limit

        self.cur = ""
        self.answer = ""
        self.msg_id: int | None = None
        self.reasoning_chars = 0
        self._last_render = 0.0
        self._last_typing = 0.0
        self._last_status = 0.0
        self._sent_any = False

    async def start(self, placeholder: str = "…") -> None:
        await self._render(placeholder, force=True)

    async def on_reasoning(self, delta_text: str) -> None:
        self.reasoning_chars += len(delta_text)
        if self.cur:
            return
        now = time.monotonic()
        if now - self._last_status < 2.0:
            return
        self._last_status = now
        await self._render(f"🤔 Thinking… {self.reasoning_chars} chars", force=True)

    async def feed(self, delta: str) -> None:
        if not delta:
            return
        self.answer += delta
        self.cur += delta
        while utf16_len(self.cur) > self.limit:
            head, self.cur = take_prefix(self.cur, self.limit)
            await self._render(head, force=True)
            self.msg_id = None
        await self._render(self.cur)

    async def finish(self) -> None:
        while utf16_len(self.cur) > self.limit:
            head, self.cur = take_prefix(self.cur, self.limit)
            await self._render(head, force=True)
            self.msg_id = None
        if self.cur:
            await self._render(self.cur, force=True)
        elif not self.answer.strip():
            hint = (
                "The model returned only its reasoning, no answer. "
                "Try rephrasing, or turn thinking off with /think off."
                if self.reasoning_chars
                else "The model returned nothing. Try again in a moment."
            )
            await self._render(hint, force=True)

    async def fail(self, message: str) -> None:
        tail = f"{self.cur}\n\n{message}".strip() if self.cur else message
        self.cur = tail
        await self._render(tail, force=True)

    async def _render(self, text: str, *, force: bool = False) -> None:
        payload = sanitize_for_telegram(text)
        if not payload:
            return
        now = time.monotonic()
        if not force and now - self._last_render < self.min_interval:
            return
        try:
            await self._typing(now)
            if self.msg_id is None:
                reply_to = self.reply_to if not self._sent_any else None
                self.msg_id = await self.tg.send_message(self.chat_id, payload, reply_to=reply_to)
                self._sent_any = True
            else:
                await self.tg.edit_message_text(self.chat_id, self.msg_id, payload)
            self._last_render = now
        except TelegramRateLimited as exc:
            log.warning("Rate limited while sending, waiting %.1fs", exc.retry_after)
            await asyncio.sleep(exc.retry_after + 0.5)
        except TelegramError as exc:
            if "not found" in exc.description or "message to edit" in exc.description:
                self.msg_id = None
            else:
                log.warning("Render failed: %s", exc)

    async def _typing(self, now: float) -> None:
        if now - self._last_typing < 4.0:
            return
        self._last_typing = now
        await self.tg.send_chat_action(self.chat_id)


class Bridge:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        *,
        telegram: TelegramClient | None = None,
        llm: ChatClient | None = None,
        handle_signals: bool = True,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.tg = telegram or TelegramClient(
            cfg.telegram_token,
            api_base=cfg.telegram_api_base,
        )
        self.ds = llm or build_client(cfg)
        self._handle_signals = handle_signals
        self.bot_id: int | None = None
        self.bot_username = ""
        self._offset: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._locks: dict[int, asyncio.Lock] = {}
        self._active_turns: dict[int, asyncio.Task] = {}
        self._stop = asyncio.Event()
        self._denied_at: dict[int, float] = {}

    # ---------- lifecycle ----------

    async def run(self) -> None:
        me = await self._connect()
        self.bot_id = int(me["id"])
        self.bot_username = str(me.get("username") or "")
        log.info("Connected as @%s (id=%s)", self.bot_username or "?", self.bot_id)
        if self.cfg.is_setup_mode:
            log.warning("ALLOWED_USER_IDS is empty: only /whoami will be answered")
        else:
            log.info("Allowed user ids: %s", ", ".join(str(i) for i in sorted(self.cfg.allowed_user_ids)))
        log.info(
            "Provider %s, model %s, thinking %s",
            self.cfg.provider,
            self.cfg.default_model,
            "on" if self.cfg.default_thinking else "off",
        )

        await self.tg.set_my_commands([{"command": c, "description": d} for c, d in COMMANDS])
        if self._handle_signals:
            self._install_signal_handlers()
        await self._skip_backlog()
        try:
            await self._poll_forever()
        finally:
            await self._shutdown()

    async def _connect(self) -> dict:
        """Retry a few times on startup instead of dumping a traceback."""
        delay = 2.0
        for attempt in range(1, 4):
            try:
                return await self.tg.get_me()
            except TelegramError as exc:
                log.error("Telegram connection failed (attempt %d/3): %s", attempt, exc)
                if attempt == 3:
                    log.error("Check TELEGRAM_BOT_TOKEN and whether this machine can reach api.telegram.org")
                    raise SystemExit(1) from exc
                await asyncio.sleep(delay)
                delay *= 2
        raise SystemExit(1)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stop.set)

    async def _skip_backlog(self) -> None:
        """offset=-1 asks for the last update only, so a restart does not replay old messages."""
        try:
            updates = await self.tg.get_updates(offset=-1, timeout=0, limit=1)
        except TelegramError as exc:
            log.warning("Could not probe for a backlog: %s", exc)
            return
        if updates:
            self._offset = int(updates[-1]["update_id"]) + 1
            log.info("Skipping the backlog, starting from update %s", self._offset)

    async def _poll_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                updates = await self.tg.get_updates(offset=self._offset, timeout=self.cfg.poll_timeout)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except TelegramError as exc:
                log.warning("Fetching updates failed: %s", exc)
                await self._sleep_or_stop(backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            for update in updates:
                self._offset = int(update["update_id"]) + 1
                task = asyncio.create_task(self._handle_safely(update))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    async def _sleep_or_stop(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)

    async def _shutdown(self) -> None:
        if self._tasks:
            log.info("Waiting for %d in-flight request(s) to finish…", len(self._tasks))
            done, pending = await asyncio.wait(self._tasks, timeout=10)
            for task in pending:
                task.cancel()
        try:
            self.store.save()
        except OSError as exc:
            log.warning("Failed to save state: %s", exc)
        await self.tg.aclose()
        await self.ds.aclose()
        log.info("Shut down")

    # ---------- dispatch ----------

    async def _handle_safely(self, update: dict) -> None:
        try:
            await self._handle_update(update)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Error while handling update %s", update.get("update_id"))

    async def _handle_update(self, update: dict) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        if sender.get("is_bot"):
            return

        chat_id = int(chat["id"])
        message_id = int(message["message_id"])
        sender_id = int(sender.get("id") or 0)
        chat_type = str(chat.get("type") or "private")
        in_group = chat_type in {"group", "supergroup"}
        # Quoting the trigger makes sense in a group; in a one-to-one chat it is
        # just a duplicated block of text above every answer.
        quote = in_group

        text = (message.get("text") or "").strip()
        caption = (message.get("caption") or "").strip()
        photo = message.get("photo")

        command, args, addressed_to = self._parse_command(text)
        if command and addressed_to and addressed_to.lower() != self.bot_username.lower():
            return
        mentioned = bool(command) or self._mentions_me(message, text or caption)

        if in_group and not mentioned:
            return

        if self.cfg.is_setup_mode:
            await self._setup_hint(chat_id, message_id, sender_id)
            return

        if sender_id not in self.cfg.allowed_user_ids:
            await self._deny(chat_id, message_id, sender_id)
            return

        if command:
            handler = {
                "start": self._cmd_help,
                "help": self._cmd_help,
                "new": self._cmd_new,
                "reset": self._cmd_new,
                "clear": self._cmd_new,
                "model": self._cmd_model,
                "think": self._cmd_think,
                "system": self._cmd_system,
                "stats": self._cmd_stats,
                "whoami": self._cmd_whoami,
            }.get(command)
            if handler is None:
                await self._reply(chat_id, message_id, f"Unknown command /{command} — try /help.")
                return
            await handler(chat_id, message_id, args, sender_id)
            return

        if photo:
            await self._handle_photo(chat_id, message_id, caption, photo, quote=quote)
            return

        cleaned = self._strip_mention(text)
        if not cleaned:
            await self._reply(chat_id, message_id, "Send some text, or see /help.")
            return
        if len(cleaned) > 8000:
            await self._reply(
                chat_id, message_id, "That message is too long (over 8000 characters). Split it up."
            )
            return
        await self._run_turn(chat_id, message_id, cleaned, quote=quote)

    # ---------- authorization helpers ----------

    async def _setup_hint(self, chat_id: int, message_id: int, sender_id: int) -> None:
        lines = [
            "This bot has no allowlist configured yet, so it can't be used.",
            "",
            f"Your user id: {sender_id}",
            "",
            "Put this in .env:",
            f"ALLOWED_USER_IDS={sender_id}",
            "",
            "Then apply it (docker compose up -d — restart doesn't re-read .env).",
        ]
        await self._reply(chat_id, message_id, "\n".join(lines))

    async def _deny(self, chat_id: int, message_id: int, sender_id: int) -> None:
        now = time.monotonic()
        if now - self._denied_at.get(sender_id, 0.0) < 300:
            return
        self._denied_at[sender_id] = now
        await self._reply(
            chat_id,
            message_id,
            f"⛔ Not authorized.\n\nYour user id: {sender_id}\n"
            "Add it to ALLOWED_USER_IDS in .env and apply the change.",
        )

    # ---------- commands ----------

    async def _cmd_help(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        await self._reply(chat_id, message_id, HELP_TEXT)

    async def _cmd_new(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        running = self._active_turns.get(chat_id)
        interrupted = running is not None and not running.done()
        if interrupted:
            running.cancel()

        state = self.store.get(chat_id)
        dropped = len(state.history)
        state.history.clear()
        self.store.save()

        if interrupted:
            await self._reply(
                chat_id, message_id, "Stopped the reply that was still generating, and cleared the context."
            )
        else:
            await self._reply(
                chat_id, message_id, f"Cleared the context ({dropped} message(s) dropped). Starting fresh."
            )

    async def _cmd_model(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        state = self.store.get(chat_id)
        current = state.model or self.cfg.default_model
        if not args:
            await self._reply(
                chat_id,
                message_id,
                "\n".join(
                    [
                        f"Current model: {current}",
                        "",
                        "deepseek-flash — fast, cheap, understands images",
                        "deepseek-v4-pro — smarter, pricier, text only",
                        "",
                        "Switch with /model flash, /model pro, or any model id your",
                        "provider offers — e.g. /model gpt-4.1",
                    ]
                ),
            )
            return

        key = args.strip()
        aliases = {
            "flash": "deepseek-flash",
            "fs": "deepseek-flash",
            "pro": "deepseek-v4-pro",
            "v4": "deepseek-v4-pro",
            "v4pro": "deepseek-v4-pro",
        }
        target = aliases.get(key.lower(), key)
        if not target or len(target) > 120 or any(ch.isspace() for ch in target):
            await self._reply(chat_id, message_id, "That does not look like a model id.")
            return
        state.model = target
        self.store.save()
        note = ""
        if target == "deepseek-v4-pro":
            note = " (this model does not accept images)"
        elif target == "deepseek-flash":
            note = " (images supported)"
        await self._reply(chat_id, message_id, f"Switched to {target}{note}")

    async def _cmd_think(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        if not self.cfg.supports_thinking:
            await self._reply(
                chat_id,
                message_id,
                f"Thinking mode is not available with PROVIDER={self.cfg.provider}.",
            )
            return
        state = self.store.get(chat_id)
        current = self.cfg.default_thinking if state.thinking is None else state.thinking
        if not args:
            target = not current
        else:
            raw = args.strip().lower()
            if raw in {"on", "1", "true", "yes"}:
                target = True
            elif raw in {"off", "0", "false", "no"}:
                target = False
            else:
                await self._reply(chat_id, message_id, "Usage: /think to toggle, or /think on and /think off")
                return
        state.thinking = target
        self.store.save()
        detail = (
            "It reasons before answering: slower and pricier, but better on hard questions."
            if target
            else "Answers come straight away, which keeps it fast."
        )
        await self._reply(chat_id, message_id, f"Thinking mode is now {'on' if target else 'off'}. {detail}")

    async def _cmd_system(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        state = self.store.get(chat_id)
        if not args:
            current = state.system_prompt or self.cfg.system_prompt
            source = "custom for this chat" if state.system_prompt else "default"
            hint = "Change it with /system <your instructions>; restore the default with /system reset"
            await self._reply(chat_id, message_id, f"System prompt ({source}):\n\n{current}\n\n{hint}")
            return
        if args.strip().lower() in {"reset", "default"}:
            state.system_prompt = None
            self.store.save()
            await self._reply(chat_id, message_id, "Restored the default system prompt.")
            return
        if len(args) > 2000:
            await self._reply(chat_id, message_id, "That prompt is too long (2000 characters max).")
            return
        state.system_prompt = args.strip()
        self.store.save()
        await self._reply(chat_id, message_id, "System prompt updated. It takes effect on your next message.")

    async def _cmd_stats(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        state = self.store.get(chat_id)
        model = state.model or self.cfg.default_model
        thinking = self.cfg.default_thinking if state.thinking is None else state.thinking
        await self._reply(
            chat_id,
            message_id,
            "\n".join(
                [
                    f"Model: {model}",
                    f"Thinking: {'on' if thinking else 'off'}",
                    f"Context: {len(state.history)} message(s)",
                    f"Turns so far: {state.total_turns}",
                    f"Tokens so far: {state.total_prompt_tokens} in / {state.total_completion_tokens} out",
                ]
            ),
        )

    async def _cmd_whoami(self, chat_id: int, message_id: int, args: str, sender_id: int) -> None:
        await self._reply(
            chat_id,
            message_id,
            f"user id: {sender_id}\nchat id: {chat_id}\n\nThe allowlist wants the user id.",
        )

    # ---------- conversation ----------

    async def _run_turn(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        image_url: str | None = None,
        quote: bool = False,
    ) -> None:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            current = asyncio.current_task()
            if current is not None:
                self._active_turns[chat_id] = current
            state = self.store.get(chat_id)
            model = state.model or self.cfg.default_model
            thinking = self.cfg.default_thinking if state.thinking is None else state.thinking
            system_prompt = state.system_prompt or self.cfg.system_prompt

            history = trim_history(
                state.history,
                self.cfg.history_max_messages,
                self.cfg.history_max_chars,
            )
            if image_url:
                user_content: str | list[dict] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
                history_text = f"[image] {text}"
            else:
                user_content = text
                history_text = text

            messages = build_messages(system_prompt, history, user_content)
            renderer = StreamRenderer(
                self.tg,
                chat_id,
                reply_to=message_id if quote else None,
                min_interval=self.cfg.edit_interval,
            )
            await renderer.start()

            usage: Usage | None = None
            try:
                async for event in self.ds.stream(
                    model=model,
                    messages=messages,
                    thinking=thinking,
                    max_tokens=self.cfg.max_output_tokens,
                ):
                    if isinstance(event, Usage):
                        usage = event
                    elif event.kind == "reasoning":
                        await renderer.on_reasoning(event.text)
                    else:
                        await renderer.feed(event.text)
            except ChatError as exc:
                log.warning("LLM request failed: %s", exc)
                await renderer.fail(f"⚠️ {exc}")
                return
            except asyncio.CancelledError:
                await renderer.fail("⚠️ Cancelled")
                raise
            except Exception as exc:
                log.exception("Error while generating a reply")
                await renderer.fail(f"⚠️ Something went wrong: {exc}")
                return

            await renderer.finish()
            answer = renderer.answer.strip()
            if not answer:
                return

            state.history.append({"role": "user", "content": history_text})
            state.history.append({"role": "assistant", "content": answer})
            state.history = trim_history(
                state.history,
                self.cfg.history_max_messages,
                self.cfg.history_max_chars,
            )
            state.total_turns += 1
            if usage:
                state.total_prompt_tokens += usage.prompt_tokens
                state.total_completion_tokens += usage.completion_tokens
            try:
                self.store.save()
            except OSError as exc:
                log.warning("Failed to save state: %s", exc)

            if usage:
                log.info(
                    "chat=%s model=%s in=%s out=%s cached=%s",
                    chat_id,
                    model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cached_tokens,
                )

    async def _handle_photo(
        self,
        chat_id: int,
        message_id: int,
        caption: str,
        photos: list[dict],
        *,
        quote: bool = False,
    ) -> None:
        state = self.store.get(chat_id)
        model = state.model or self.cfg.default_model
        if model == "deepseek-v4-pro":
            await self._reply(
                chat_id,
                message_id,
                f"{model} doesn't accept images. Switch with /model flash, then send the photo again.",
            )
            return

        largest = photos[-1]
        try:
            info = await self.tg.get_file(str(largest["file_id"]))
            data = await self.tg.download(str(info["file_path"]))
        except TelegramError as exc:
            await self._reply(chat_id, message_id, f"Could not download the image: {exc}")
            return

        if len(data) > _MAX_IMAGE_BYTES:
            await self._reply(
                chat_id, message_id, "That image is too large (over 20 MB). Compress it and try again."
            )
            return
        mime = sniff_image_mime(data)
        if mime is None:
            await self._reply(chat_id, message_id, "Unsupported image format. JPEG, PNG, GIF and WebP only.")
            return

        encoded = base64.b64encode(data).decode("ascii")
        prompt = caption.strip() or "Describe what is in this image."
        log.info("chat=%s received an image, %.1f KB (%s)", chat_id, len(data) / 1024, mime)
        await self._run_turn(
            chat_id,
            message_id,
            prompt,
            image_url=f"data:{mime};base64,{encoded}",
            quote=quote,
        )

    # ---------- small helpers ----------

    async def _reply(self, chat_id: int, message_id: int | None, text: str) -> None:
        first = True
        for chunk in split_text(text):
            try:
                await self.tg.send_message(chat_id, chunk, reply_to=message_id if first else None)
            except TelegramError as exc:
                log.warning("Reply failed: %s", exc)
                return
            first = False

    def _parse_command(self, text: str) -> tuple[str | None, str, str | None]:
        match = _COMMAND_RE.match(text or "")
        if not match:
            return None, "", None
        return match.group(1).lower(), match.group(3).strip(), match.group(2)

    def _mentions_me(self, message: dict, text: str) -> bool:
        reply_to = message.get("reply_to_message") or {}
        if int((reply_to.get("from") or {}).get("id") or 0) == self.bot_id:
            return True
        if not self.bot_username or not text:
            return False
        return f"@{self.bot_username.lower()}" in text.lower()

    def _strip_mention(self, text: str) -> str:
        if not self.bot_username:
            return text.strip()
        pattern = re.compile(re.escape(f"@{self.bot_username}"), re.IGNORECASE)
        return pattern.sub("", text).strip()
