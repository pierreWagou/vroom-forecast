FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the serving sub-project
COPY serving/ serving/

# Install deps (Ray Serve + FastAPI + ML deps)
RUN uv sync --project serving

# Ray Serve HTTP port + Ray dashboard
EXPOSE 8000 8265

CMD ["uv", "run", "--project", "serving", "python", "-m", "serving"]
