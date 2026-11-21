#!/usr/bin/env bash
# Raspberry Pi / Linux deploy helper: check the box, create .env, build and start.
# Usage: bash deploy/pi-setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

info() { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
die() {
    printf '\033[1;31m✗\033[0m %s\n' "$*" >&2
    exit 1
}

# ---------- 1. Architecture ----------
arch=$(uname -m)
case "$arch" in
    aarch64 | arm64)
        info "Architecture ${arch} — good"
        ;;
    armv7l)
        warn "32-bit system (armv7l). Docker may not pull the image; use 64-bit Raspberry Pi OS."
        ;;
    *)
        die "Unsupported architecture: ${arch}"
        ;;
esac

# ---------- 2. Memory ----------
mem_mb=$(awk '/MemTotal/ {printf "%d", $2 / 1024}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$mem_mb" -gt 0 ]; then
    [ "$mem_mb" -ge 800 ] || warn "Only ${mem_mb} MB of RAM available; the build may fail."
    info "Memory ${mem_mb} MB (the bot needs about 60 MB once running)"
fi

# ---------- 3. Docker ----------
command -v docker >/dev/null 2>&1 || die "Docker is not installed. Run:
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker \$USER
  then log out and back in (or run 'newgrp docker') and try again."

if docker compose version >/dev/null 2>&1; then
    DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    DC=(docker-compose)
else
    die "Docker is installed but the compose plugin is missing:
    sudo apt install docker-compose-plugin"
fi
info "Using ${DC[*]}"

docker info >/dev/null 2>&1 || die "This user has no Docker permission:
    sudo usermod -aG docker \$USER
  then log out and back in."

# ---------- 4. .env ----------
if [ ! -f .env ]; then
    cp .env.example .env
    info "Created .env"
    die "Open .env, fill in TELEGRAM_BOT_TOKEN and API_KEY, then run this script again."
fi

token=$(grep -E '^TELEGRAM_BOT_TOKEN=' .env | cut -d= -f2- | tr -d '[:space:]' || true)
key=$(grep -E '^(API_KEY|DEEPSEEK_API_KEY)=' .env | head -1 | cut -d= -f2- | tr -d '[:space:]' || true)
[ -n "$token" ] || die "TELEGRAM_BOT_TOKEN is still empty in .env"
[ -n "$key" ] || die "API_KEY is still empty in .env"
case "$token" in
    *:*) ;;
    *) warn "TELEGRAM_BOT_TOKEN has no colon — it should look like 123456:AAH..." ;;
esac

if grep -qE '^ALLOWED_USER_IDS=[[:space:]]*$' .env; then
    info "Allowlist is empty — start the bot, then send it /whoami to learn your id"
fi

# ---------- 5. Disk ----------
free_mb=$(df -Pm . | awk 'NR==2 {print $4}')
[ "$free_mb" -ge 1500 ] || warn "Only ${free_mb} MB free; building the image needs about 1 GB."
info "Disk free ${free_mb} MB"

# ---------- 6. Build and start ----------
info "Building the image (a minute or two the first time)..."
"${DC[@]}" up -d --build

info "Waiting for it to reach Telegram..."
sleep 6
log_tail=$("${DC[@]}" logs --tail 40 2>&1 || true)
printf '%s\n' "$log_tail"

case "$log_tail" in
    *PermissionError*)
        warn "File permissions inside the image are too strict. Rebuild:"
        warn "  ${DC[*]} build --no-cache && ${DC[*]} up -d"
        ;;
    *Unauthorized*)
        warn "TELEGRAM_BOT_TOKEN is wrong: check for stray spaces or a truncated copy"
        ;;
    *"out of credit"* | *insufficient* | *402*)
        warn "DeepSeek has no balance left — top up at platform.deepseek.com"
        ;;
esac

cat <<EOF

────────────────────────────────────────
If the line above says "Connected as @your_bot", it worked.

Next: send /whoami to the bot from Telegram,
then put the id it returns into .env:

    ALLOWED_USER_IDS=your-id

Apply the new configuration:

    ${DC[*]} up -d

Follow the logs:

    ${DC[*]} logs -f
────────────────────────────────────────
EOF
