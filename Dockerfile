FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ✅ FIX: copy entire project
COPY . /app

RUN mkdir -p /app/data
VOLUME ["/app/data"]

RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

ENV PYTHONPATH=/app

WORKDIR /app/data

CMD ["python", "/app/main.py"]