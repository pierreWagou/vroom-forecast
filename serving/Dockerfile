FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY serving/ serving/

# Install deps
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "serving"]
