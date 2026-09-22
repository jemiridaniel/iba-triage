# syntax=docker/dockerfile:1

# ---- 1. Build the PWA ----------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# ---- 2. Python runtime ---------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
# uv and its download cache are mounted for this step only, so neither ends up in a layer.
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.17,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY backend/ backend/
COPY data/sources.yaml data/endemicity.yaml data/
COPY data/mock/ data/mock/
COPY --from=frontend /app/frontend/dist frontend/dist

# Non-root. The guideline index (data/index) is mounted at runtime, not baked in:
# NCDC documents have no stated licence, so their text is not redistributed in the image.
RUN useradd --system --uid 10001 --no-create-home iba \
    && mkdir -p /app/.cache /app/data/index \
    && chown -R iba /app/.cache /app/data/index
USER iba

ENV PATH="/app/.venv/bin:$PATH" \
    APP_ENV=prod \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)"

CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
