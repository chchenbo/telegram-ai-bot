"""Persist per-chat context to disk so a restart does not lose the conversation."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ChatState:
    history: list[dict] = field(default_factory=list)
    model: str | None = None
    thinking: bool | None = None
    system_prompt: str | None = None
    total_turns: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "history": self.history,
            "model": self.model,
            "thinking": self.thinking,
            "system_prompt": self.system_prompt,
            "total_turns": self.total_turns,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatState:
        history = data.get("history")
        if not isinstance(history, list):
            history = []
        clean = [
            {"role": str(m.get("role")), "content": str(m.get("content") or "")}
            for m in history
            if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
        ]
        return cls(
            history=clean,
            model=data.get("model"),
            thinking=data.get("thinking"),
            system_prompt=data.get("system_prompt"),
            total_turns=int(data.get("total_turns") or 0),
            total_prompt_tokens=int(data.get("total_prompt_tokens") or 0),
            total_completion_tokens=int(data.get("total_completion_tokens") or 0),
        )


class Store:
    """All chats in one JSON file, written via temp file + rename so a power cut cannot corrupt it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.chats: dict[int, ChatState] = {}

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            backup = self.path.with_suffix(self.path.suffix + ".broken")
            log.warning("State file is corrupt (%s); moved to %s and starting fresh", exc, backup)
            with contextlib.suppress(OSError):
                os.replace(self.path, backup)
            return

        chats = raw.get("chats") if isinstance(raw, dict) else None
        if not isinstance(chats, dict):
            return
        for key, value in chats.items():
            if not isinstance(value, dict):
                continue
            try:
                self.chats[int(key)] = ChatState.from_dict(value)
            except (TypeError, ValueError):
                continue
        log.info("Loaded context for %d chat(s)", len(self.chats))

    def get(self, chat_id: int) -> ChatState:
        state = self.chats.get(chat_id)
        if state is None:
            state = ChatState()
            self.chats[chat_id] = state
        return state

    def save(self) -> None:
        payload = {"version": 1, "chats": {str(k): v.to_dict() for k, v in self.chats.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
