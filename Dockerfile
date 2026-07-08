FROM python:3.11-slim

<<<<<<< HEAD
=======
# ── System setup ─────────────────────────────────────────────────────────────
>>>>>>> 162b07b (push full code)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

<<<<<<< HEAD
=======
# Build tools for numpy/pandas wheels; procps for the compose healthcheck (pgrep)
>>>>>>> 162b07b (push full code)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

<<<<<<< HEAD
=======
# ── Python deps (cached layer) ───────────────────────────────────────────────
>>>>>>> 162b07b (push full code)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

<<<<<<< HEAD
# ✅ FIX: copy entire project
COPY . /app

RUN mkdir -p /app/data
VOLUME ["/app/data"]

RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

ENV PYTHONPATH=/app

WORKDIR /app/data

CMD ["python", "/app/main.py"]
=======
# ── App code ─────────────────────────────────────────────────────────────────
# indicators.py MUST be copied alongside the bot (it's imported as `ta`)
COPY indicators.py .
COPY crypto_bot_telegram.py .

# Persisted output (trades.csv + bot.log) lives here; mount a volume on it
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# Run as non-root for safety
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# The bot writes trades.csv / bot.log to the working dir; point it at /app/data
WORKDIR /app/data

CMD ["python", "/app/crypto_bot_telegram.py"]
>>>>>>> 162b07b (push full code)
