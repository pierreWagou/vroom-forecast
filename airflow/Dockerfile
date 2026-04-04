FROM apache/airflow:2.10.5-python3.12

# Silence SyntaxWarnings from bundled provider packages (azure, etc.)
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

# Install uv — used at runtime to execute pipelines in isolated envs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy each sub-project with its own pyproject.toml + uv.lock + source.
# Airflow runs training, promotion, and features as independent uv projects.
COPY training/ training/
COPY promotion/ promotion/
COPY features/ features/
