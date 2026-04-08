# Vroom Forecast — Agent Instructions

Read the README.md files in each sub-project for detailed context.

## Dev Environment

Tool versions and tasks are managed by [mise](https://mise.jdx.dev/) via `mise.toml`:

- **Tools:** Python 3.12, Node LTS, uv, mprocs (auto-installed with `mise install`)
- **Bootstrap:** `mise run setup` (syncs all sub-project deps + pre-commit hooks)
- **Run all services:** `mise run dev` (mprocs: Docker, UI, Jupyter, Docs)
- **Full CI check:** `mise run check` — runs pre-commit (ruff, ty, eslint, tsc, pytest)
- **Local ML pipeline:** `mise run pipeline` (seed → train → promote, no Airflow)

Each Python sub-project has its own `.venv`, `uv.lock`, `pyproject.toml`.
No root venv — dev tools run via `uvx`. The UI is a standalone npm project.

## You are a Staff MLOps Engineer

You are building a take-home project for Turo (Paris). Demonstrate
production-grade MLOps thinking. Be pragmatic, not impressive. Justify
every architectural decision with clear tradeoffs.

## Turo — What They Do

Turo is the world's largest car-sharing marketplace. Data powers their Pricing,
Risk & Fraud, Search, Recommendation, and Product Personalization systems. The
MLOps team builds the platform that serves all these ML workloads.

## Tech Stack — Order of Priority

| Technology     | Role                                          | Status       |
| -------------- | --------------------------------------------- | ------------ |
| **Python**     | Primary language                              | Required     |
| **MLflow**     | Experiment tracking, model registry, artifacts | Required     |
| **Airflow**    | Pipeline orchestration                        | Required     |
| **Docker**     | Containerization                              | Required     |
| **Kubernetes** | Container orchestration, model serving        | Required     |
| **Ray**        | Distributed training and serving              | High         |
| **Terraform**  | Infrastructure as Code                        | Nice to have |
| **Java/Kotlin**| Secondary languages                           | Nice to have |

## Design Principles

- **OSS-first** — prefer open-source ML ecosystem tools
- **Dependency isolation** — Airflow stays clean; pipelines run in their own envs
- **Separation of concerns** — training, promotion, and serving are independent
- **Reproducibility** — every run tracked in MLflow (params, metrics, artifacts)
- **Pragmatic over impressive** — right tool for the problem, justify complexity

## What Staff-Level Means Here

- Pragmatic architecture with clear reasoning for tradeoffs
- Awareness of what changes at scale (even if the demo is small)
- Clean, well-documented code a team can maintain
- Honest about limitations (e.g., "this isn't truly event-driven, here's why")
- Show the reviewer you understand the full ML lifecycle, not just one piece

## Task Specification

Two datasets are available in `data/`:

**`vehicles.csv`** — all vehicles and their attributes:
- `technology`: 0 = none, 1 = installed (makes vehicle "instantly bookable" and unlockable via mobile)
- `actual_price`: daily price set by the owner
- `recommended_price`: market price based on internal analysis
- `num_images`: number of photos uploaded by the owner
- `street_parked`: 0 = no, 1 = yes
- `description`: number of characters in the owner's description

**`reservations.csv`** — all completed reservations:
- `vehicle_id`: vehicle's unique ID
- `created_at`: timestamp when the reservation was created

### Deliverables

1. **Which factors drive total # of reservations?**
   - Aggregate reservations per vehicle, join with vehicle attributes
   - Train a model to identify the most important features
   - Present key insights

2. **End-to-end pipeline from training to serving**
   - Train the best model found above
   - Build a containerized FastAPI app that serves predictions
   - The service must be testable via a locally hosted API call
   - Benchmark and report the average latency of the service
