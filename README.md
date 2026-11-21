# Talk to any LLM from Telegram

A small self-hosted bot that turns a Telegram chat into a front end for **DeepSeek, OpenAI or Claude**. Streaming answers, conversation memory, image understanding, model switching.

![tests](https://github.com/chchenbo/telegram-ai-bot/actions/workflows/ci.yml/badge.svg)

## What you get

- **Streaming replies** that fill in as they arrive, not one blob at the end
- **Memory per chat** — it keeps the thread; `/new` wipes it
- **Photos work** — send an image and the model looks at it
- **Switch models mid-conversation** with `/model`, and let DeepSeek or Claude think first with `/think`
- **Only you get answers** — a numeric allowlist checked on every message
- **Long answers split themselves**, measured in UTF-16 so emoji never break mid-character
- **No inbound ports.** It polls Telegram outbound, so it runs happily behind NAT

## Providers

| Provider | Protocol | Images | Thinking |
|---|---|---|---|
| DeepSeek | OpenAI-compatible | yes | yes |
| OpenAI | OpenAI | yes | no switch |
| Anthropic | native `/v1/messages` | yes | yes |

Switching provider is a couple of lines in `.env` — no code changes.

## Install

```bash
git clone https://github.com/chchenbo/telegram-ai-bot && cd telegram-ai-bot
curl -fsSL https://get.docker.com | sh    # if Docker isn't installed yet
bash deploy/pi-setup.sh                   # checks the box, creates .env, builds
```

Fill in `TELEGRAM_BOT_TOKEN` and `API_KEY` when it stops, run it again, then send `/whoami` to the bot and put the number it returns into `ALLOWED_USER_IDS`.

**Configuration, costs, architecture and troubleshooting: [GUIDE.md](GUIDE.md).**

## License

MIT
