# Guide

Everything the [README](README.md) leaves out: hardware, install, configuration, costs, architecture, troubleshooting.

## Requirements

- A Linux box with Docker (the reference deployment is a Raspberry Pi 4B, but anything works)
- 64-bit OS — `uname -m` should print `aarch64` or `x86_64`
- A Telegram account, to create the bot
- An API key from one of the supported providers

## The reference setup

| | |
|---|---|
| Board | Raspberry Pi 4B |
| RAM | 4 GB |
| OS | Raspberry Pi OS 64-bit (`aarch64`) |
| Runtime | Docker Compose, starts on boot |
| Network | Home broadband — **no public IP, no open ports** |
| Access | SSH over LAN and Tailscale |

No ports are opened because none are needed: the bot polls Telegram outbound, so sitting behind NAT is fine.

Measured footprint:

| Metric | Value | Note |
|---|---|---|
| Resident memory | **~41 MB** | RSS after importing everything and initializing both HTTP clients |
| Idle CPU | ~0 | one long-poll request is held open every 30 s, then it sleeps |
| Image size | ~140 MB | mostly the `python:3.12-slim` base |
| Cold start | ~1 s | container start to "connected" in the logs |
| Time to first token | 1–3 s | thinking off; add 3–20 s with thinking on |
| Max message length | 3,800 UTF-16 units | longer replies are split, measured in UTF-16 so emoji survive |
| Context window | 40 messages / 48k chars | trimmed by both limits, configurable |

## Install

### 1. Create the bot

Open **https://t.me/BotFather** — follow the link rather than searching, lookalike accounts mimic it. Send `/newbot`, pick a display name, then a username ending in `bot`. It replies with a block starting "Done! Congratulations on your new bot" and a token like:

```
123456789:AAH...
```

### 2. Get a provider key

DeepSeek, OpenAI or Anthropic — whichever you picked. Keep it somewhere for the next step.

### 3. Run the setup helper

```bash
# 1. Docker (once)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
# log out and back in so the group takes effect

# 2. get the code onto the box
git clone https://github.com/chchenbo/telegram-ai-bot && cd telegram-ai-bot
#    or, from your laptop: rsync -av telegram-ai-bot pi@your-pi:~/

# 3. run the helper — it checks arch, RAM, disk, Docker permissions and .env
bash deploy/pi-setup.sh
```

The first run generates `.env` and stops. Fill in two values and run it again:

```
TELEGRAM_BOT_TOKEN=123456789:AAH...
API_KEY=sk-...
```

You are up when you see:

```
INFO ai_bot.bot: Connected as @your_bot (id=...)
```

### 4. Let yourself in

Send `/whoami` to the bot from your phone. It answers with a number; put it into `ALLOWED_USER_IDS` and apply the change:

```bash
docker compose up -d
```

> With an empty allowlist the bot runs in setup mode: it answers `/whoami` and ignores everything else, so nobody can find it and burn your credits before you finish.

Prefer systemd over Docker? `deploy/ai-bot.service` is a ready-made unit — create the
`ai-bot` user, copy the file to `/etc/systemd/system/`, adjust the paths, then
`systemctl enable --now ai-bot`.

## Configure

Everything lives in `.env` beside the code.

### Provider, key and model

Three values describe your model backend:

```
PROVIDER=deepseek                              # deepseek | openai | anthropic
API_KEY=sk-...                                 # that provider's key
DEFAULT_MODEL=deepseek-flash                   # whatever it calls the model
# API_BASE_URL=                                # only for a proxy; defaults per provider
```

`API_BASE_URL` can be omitted — each provider has a sensible default:

| `PROVIDER` | Default base URL | Thinking |
|---|---|---|
| `deepseek` | `https://api.deepseek.com` | `thinking` + `reasoning_effort` |
| `openai` | `https://api.openai.com/v1` | the model decides; `/think` says so |
| `anthropic` | `https://api.anthropic.com` | `thinking` with a token budget |

An explicit `DEFAULT_MODEL` is required for anything other than DeepSeek — there is no model name worth guessing.

`DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` still work as aliases, if you already have a `.env` from an earlier version.

### Allowlist

`ALLOWED_USER_IDS` is the only thing between a stranger and your credits. It takes **numeric Telegram user ids**, not usernames — usernames can change, ids cannot. Several people: `111111,222222`.

### Model and thinking at runtime

| Setting | Default | Trade-off |
|---|---|---|
| `DEFAULT_MODEL` | `deepseek-flash` | fast and cheap, accepts images |
| `DEFAULT_MODEL=deepseek-v4-pro` | — | smarter, roughly 3× the price, text only |
| `DEFAULT_THINKING` | `false` | `true` reasons first: better on hard questions, slower and pricier |

Both can be overridden per chat with `/model` and `/think`, and those choices live in `data/state.json`.

`/model` takes **any model id your provider accepts**, not just the two DeepSeek aliases: `/model gpt-4.1`, `/model claude-sonnet-4-5`, whatever you like. A typo reaches the API and comes back as a 404 with the provider's message.

### Every variable

| Variable | Default | |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | required | from @BotFather |
| `API_KEY` | required | provider key (`DEEPSEEK_API_KEY` also accepted) |
| `ALLOWED_USER_IDS` | empty | comma-separated numeric ids; empty = setup mode |
| `PROVIDER` | `deepseek` | `deepseek`, `openai` or `anthropic` |
| `API_BASE_URL` | provider default | override only for proxies or self-hosted endpoints |
| `DEFAULT_MODEL` | `deepseek-flash` | required for non-DeepSeek providers |
| `DEFAULT_THINKING` | `false` | DeepSeek and Claude; ignored on OpenAI |
| `SYSTEM_PROMPT` | built-in | leave empty for the default; overridable per chat |
| `HISTORY_MAX_MESSAGES` | `40` | context: max messages kept |
| `HISTORY_MAX_CHARS` | `48000` | context: max characters kept |
| `MAX_OUTPUT_TOKENS` | empty | leave empty to use the provider's default |
| `EDIT_INTERVAL` | `1.2` | seconds between streaming edits; below 0.5 you get rate-limited |
| `LOG_LEVEL` | `INFO` | `DEBUG` is chattier |

### Applying changes

Environment variables are baked into the container when it is created:

```bash
docker compose up -d     # recreates the container and re-reads .env
```

`docker compose restart` does **not** pick up a new `.env` — a classic way to lose twenty minutes. Per-chat settings (`/model`, `/think`, `/system`) apply on the next message and need no restart at all.

## Updating

```bash
git pull                        # or rsync the folder again from your laptop
docker compose up -d --build
```

`data/` and `.env` are untouched, so context and settings survive. Skipping `--build` is the most common "my update did nothing": the old image keeps running.

## Commands

| Command | What it does |
|---|---|
| `/new` | Clear the conversation; also interrupts a reply that is still generating |
| `/model` | Show the current model, or switch to any model id |
| `/think` | Toggle thinking mode (DeepSeek and Claude) |
| `/system` | Show or set the system prompt; `/system reset` restores the default |
| `/stats` | Context size, turn count, token totals |
| `/whoami` | Your user id and chat id |
| `/help` | Usage |

In groups the bot stays quiet unless you mention it or reply to it; it also quotes the message it answers there. In a one-to-one chat it never quotes. Commands work without a mention.

## Cost

Cost depends on the provider; this table is DeepSeek, because that's what the reference deployment uses. USD per 1M tokens, list price as of Nov 2026:

| Model | Input (cache miss) | Output | Images |
|---|---|---|---|
| `deepseek-flash` | $0.15 off-peak / $0.30 peak | $0.60 / $1.20 | yes |
| `deepseek-v4-pro` | $0.66 / $1.32 | $1.98 / $3.96 | no |

Peak hours are 01:00–04:00 and 06:00–10:00 UTC, Monday to Friday; everything else is half price. A 500-character answer on `deepseek-flash` off-peak costs roughly $0.0004.

## How it works

```
phone (Telegram) ──long poll──> container on the Pi ──HTTPS──> the provider's API
                                      │
                                      ├─ allowlist check
                                      ├─ context assembly (trimmed by count and size)
                                      └─ streaming render: edit one message, start a new one when full
```

A few deliberate choices:

- **Long polling instead of webhooks.** No domain, no certificate, no inbound port.
- **Streaming via `editMessageText`.** Telegram rate-limits edits, so it redraws every 1.2 s and backs off on HTTP 429.
- **Images only in the current turn.** History keeps an `[image] question` placeholder, otherwise token usage explodes after a few photos.
- **Backlog is skipped on startup.** Otherwise a power cut would replay every question from the last few hours.
- **Provider quirks isolated in one place.** `PROVIDER` decides which protocol and which fields go on the wire; the rest of the code doesn't care who answers.

### Layout

```
src/ai_bot/
  bot.py        polling, allowlist, commands, streaming renderer
  llm.py        ChatClient (DeepSeek/OpenAI) + AnthropicClient
  config.py     .env parsing and validation
  telegram.py   minimal Bot API wrapper
  store.py      per-chat context persisted to JSON
  textutil.py   UTF-16 splitting and history trimming
tests/          74 tests, including an end-to-end smoke suite
deploy/         setup script and systemd unit
```

## Privacy and data

- Conversations are stored **in plain text** in `data/state.json` on your own machine. Anyone with shell access to that box can read them.
- Messages go to whichever provider you configured; that provider can read them. The bot adds no middleman of its own.
- The bot token and API key live in `.env`. Both are excluded from git, and neither is encrypted at rest.
- The allowlist is the only access control, and it matches numeric user ids — not usernames, which can change hands.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `400 ... Unrecognized request argument: thinking` | the endpoint rejects DeepSeek's extensions — set `PROVIDER=openai` |
| `404 model not found` | wrong `DEFAULT_MODEL` for that provider |
| `PermissionError: /app/src/...` | files in the image are too restrictive; `docker compose build --no-cache` |
| `getMe: Unauthorized` | wrong bot token, or a stray space |
| cannot reach `api.telegram.org` | the network blocks Telegram — set `HTTPS_PROXY` in `.env` |
| `401` | wrong `API_KEY` for that `API_BASE_URL` |
| `402` / "out of credit" | the provider account has no balance |
| `.env` changes ignored | `restart` reuses the old environment; use `compose up -d` |
| Old messages unanswered after a restart | intended — the backlog is skipped |

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

74 cases covering UTF-16 splitting boundaries, SSE parsing, config validation, provider payloads, corrupt-state recovery, plus an end-to-end smoke suite that drives the real polling loop against a fake Telegram API and a fake model stream (auth, group mentions, context carry-over, commands, interrupting a reply).
