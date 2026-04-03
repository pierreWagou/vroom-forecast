FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the serving sub-project (pyproject.toml, lock, source)
COPY serving/ serving/

# Install deps
RUN uv sync --project serving

EXPOSE 8000

CMD ["uv", "run", "--project", "serving", "python", "-m", "serving"]
