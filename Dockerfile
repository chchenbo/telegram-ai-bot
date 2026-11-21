FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# COPY preserves host permission bits. Loosen them here, otherwise a non-root
# container user cannot read its own source (PermissionError: /app/src/...).
RUN chmod -R a+rX /app/src

RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /data \
    && chown -R bot:bot /data

USER bot

ENV STATE_PATH=/data/state.json
VOLUME ["/data"]

CMD ["python", "-m", "ai_bot"]
