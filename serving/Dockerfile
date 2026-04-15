FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

# Copy the serving sub-project
COPY serving/ serving/

# Install deps (Ray Serve + FastAPI + ML deps)
RUN uv sync --project serving

# Ray Serve HTTP port + Ray dashboard
EXPOSE 8000 8265

# Activate the venv directly instead of `uv run` so that VIRTUAL_ENV
# matches the path Ray's raylet resolves. The mismatch between
# uv's absolute /app/serving/.venv and Ray's relative serving/.venv
# caused the controller to crash during logging setup.
ENV VIRTUAL_ENV=/app/serving/.venv
ENV PATH="/app/serving/.venv/bin:$PATH"
# Log to stderr instead of files — avoids a Ray 2.49 bug where the
# MemoryHandler's target is None during controller recovery, crashing
# on handler.target.baseFilename.
ENV RAY_SERVE_LOG_TO_STDERR=1

CMD ["python", "-m", "serving"]
