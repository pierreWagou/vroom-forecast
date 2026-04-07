FROM apache/airflow:2.10.5-python3.12

# Silence SyntaxWarnings from bundled provider packages (azure, etc.)
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

# Install uv — used at runtime to execute pipelines in isolated envs
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

# Copy each sub-project with its own pyproject.toml + uv.lock + source.
# Airflow runs training, promotion, and features as independent uv projects.
# --chown=airflow ensures the airflow user can create .venv dirs inside them.
COPY --chown=airflow training/ training/
COPY --chown=airflow promotion/ promotion/
COPY --chown=airflow features/ features/

# Pre-install deps so the first DAG run is fast (not waiting for uv to resolve).
RUN uv sync --project training --no-dev \
    && uv sync --project promotion --no-dev \
    && cd features && uv sync --no-dev
