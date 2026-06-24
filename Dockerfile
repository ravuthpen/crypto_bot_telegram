FROM python:3.11-slim

# ── System setup ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Build tools for numpy/pandas wheels; procps for the compose healthcheck (pgrep)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────
COPY . /app

# Persisted output (trades.csv + logs) lives here; mount a volume on it
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# Run as non-root for safety
RUN