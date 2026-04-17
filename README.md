<p align="center">
  <img src="docs/assets/hero.svg" alt="Vroom Forecast" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/pierreWagou/vroom-forecast/actions/workflows/ci.yml"><img src="https://github.com/pierreWagou/vroom-forecast/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/pierreWagou/vroom-forecast/actions/workflows/cd-docs.yml"><img src="https://github.com/pierreWagou/vroom-forecast/actions/workflows/cd-docs.yml/badge.svg" alt="CD"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"/></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/Ray_Serve-028CF0?logo=ray&logoColor=white" alt="Ray Serve"/>
  <img src="https://img.shields.io/badge/Airflow-017CEE?logo=apacheairflow&logoColor=white" alt="Airflow"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Next.js-000000?logo=next.js&logoColor=white" alt="Next.js"/>
</p>

---

ML pipeline that predicts the number of reservations a vehicle will receive
based on its listing attributes. Built as a take-home project for a
**Staff MLOps Engineer** position at [Turo](https://turo.com).

> **[Live documentation](https://pierrewagou.github.io/vroom-forecast/)** &nbsp;&middot;&nbsp; **[Interactive API docs](http://localhost:8000/docs)** (local)

---

## Getting Started

### Prerequisites

| Tool | Purpose |
|------|---------|
| [mise](https://mise.jdx.dev/getting-started.html) | Installs and manages all dev tools automatically |
| [Docker Desktop](https://docs.docker.com/get-docker/) | Runs MLflow, Redis, Airflow, and Ray Serve |

### Setup

```bash
git clone https://github.com/pierreWagou/vroom-forecast.git
cd vroom-forecast

# 1. Install tools (Python 3.12, Node LTS, uv, mprocs)
mise install

# 2. Bootstrap all dependencies
mise run setup

# 3. Start everything
mise run dev
```

The UI is at [localhost:3000](http://localhost:3000). Trigger the ML pipeline
from Airflow at [localhost:8080](http://localhost:8080) (credentials: `admin` / `admin`).

### Quick Test

Once services are running and the pipeline has completed (`mise run pipeline` or
trigger from Airflow):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"technology":1,"actual_price":45,"recommended_price":50,"num_images":8,"street_parked":0,"description":250}'
```

```json
{"predicted_reservations": 4.12, "model_version": "1"}
```

Interactive docs at [localhost:8000/docs](http://localhost:8000/docs).
A [Bruno](https://www.usebruno.com/) API collection is included in `bruno/` -- open it to hit every endpoint with pre-filled sample payloads.

---

## Architecture

```mermaid
graph TB
    subgraph Frontend
        UI[Next.js + shadcn/ui<br/><i>TypeScript, Tailwind</i>]
    end

    subgraph Serving["Serving — Ray Serve"]
        ING[VroomForecastApp<br/><i>FastAPI ingress</i>]
        FC[FeatureComputer<br/><i>pandas</i>]
        FL[FeatureLookup<br/><i>Feast client</i>]
        OFR[OfflineFeatureReader<br/><i>pandas / Parquet</i>]
        PRED[Predictor<br/><i>scikit-learn / MLflow</i>]
        MAT[FeatureMaterializer<br/><i>Feast / Redis pub-sub</i>]
        MRL[ModelReloadListener<br/><i>Redis pub-sub</i>]
    end

    subgraph Orchestration
        AF[Airflow<br/><i>Docker, BashOperator + uv</i>]
    end

    subgraph Storage
        DB[(SQLite<br/><i>vehicle persistence</i>)]
        PQ[(Parquet<br/><i>offline feature store</i>)]
        RD[(Redis<br/><i>online feature store + pub-sub</i>)]
        MLF[(MLflow<br/><i>model registry + tracking</i>)]
    end

    subgraph ML Pipelines["ML Pipelines — Python (uv)"]
        FEAT[features/<br/><i>Feast, pandas</i>]
        TRAIN[training/<br/><i>scikit-learn, MLflow</i>]
        PROMO[promotion/<br/><i>MLflow, Redis</i>]
    end

    UI --- ING
    ING --- FC & FL & OFR & PRED
    MAT --- RD & DB
    MRL --- RD & PRED
    FL --- RD
    OFR --- PQ
    PRED --- MLF
    AF --- FEAT & TRAIN & PROMO
    FEAT --- DB & PQ & RD
    TRAIN --- PQ & MLF
    PROMO --- MLF & RD
    ING --- DB
```

> **Zoom in:** [Feature Store](features/) · [Serving](serving/) · [Training](training/) · [Promotion](promotion/) · [Orchestration](airflow/) · [Frontend](ui/) · [Exploration](exploration/)

---

## ML Pipeline

```mermaid
sequenceDiagram
    participant Seed as seed.py
    participant DB as SQLite
    participant FP as Feature Pipeline
    participant PQ as Parquet
    participant RD as Redis
    participant TR as Training
    participant MLF as MLflow
    participant PR as Promotion
    participant SRV as Serving

    Note over Seed,DB: Daily (Airflow)
    Seed->>DB: Load CSVs (idempotent)
    FP->>DB: Read all vehicles + reservations
    FP->>FP: Compute price_diff
    FP->>PQ: Write offline store (fleet vehicles)
    FP->>RD: Write online store (new arrivals only)

    Note over TR,MLF: On fresh features (Airflow Dataset)
    TR->>PQ: Read training data
    TR->>TR: Train RandomForest (5-fold CV)
    TR->>MLF: Log metrics + register model
    TR->>MLF: Set "candidate" alias

    Note over PR,SRV: Event-driven
    PR->>MLF: Compare candidate vs champion
    PR->>MLF: Set "champion" alias (if better)
    PR->>RD: Publish "model-promoted" event
    RD->>SRV: Predictor reloads champion model
```

> **Deep dive:** [Feature pipeline](features/) · [Training](training/) · [Promotion](promotion/) · [Airflow DAGs](airflow/) · [Serving](serving/)

---

## Key Findings -- What Drives Reservations?

**Dataset:** 1,000 vehicles, 6,376 reservations. Average per vehicle: 6.4 (median 5, std 4.9).

### Production Model (5 features)

Based on the [exploration notebook](exploration/), the production model uses 5 features:
`technology`, `num_images`, `street_parked`, `description`, `price_diff`.

| Metric | Value |
|--------|-------|
| CV MAE (5-fold) | **3.44** (+/- 0.50) |
| Model | RandomForestRegressor (100 trees) |

<details>
<summary><b>Exploratory Analysis (8 features)</b></summary>

The exploration notebook tested all 8 raw and derived features:

| Rank | Feature | Importance | Correlation | Interpretation |
|------|---------|------------|-------------|----------------|
| 1 | **price_diff** | 26.2% | -0.367 | Gap between actual and recommended price. Vehicles priced **below** recommended get more reservations. |
| 2 | **price_ratio** | 20.7% | -0.398 | Confirms the pricing story. Ratio below 1.0 drives bookings. |
| 3 | **description** | 16.9% | +0.016 | Non-linear -- very short and very long descriptions both underperform. |
| 4 | **actual_price** | 11.9% | -0.259 | Lower absolute price drives reservations. |
| 5 | **num_images** | 10.5% | +0.220 | More photos, more reservations. |
| 6 | **recommended_price** | 10.1% | -0.013 | Matters only in combination with actual price. |
| 7 | **street_parked** | 2.1% | -0.017 | Minimal impact. |
| 8 | **technology** | 1.5% | +0.136 | Slight positive effect. |

Three features were dropped for production:
- `actual_price` and `recommended_price` -- collinear with `price_diff`
- `price_ratio` -- 91% correlated with `price_diff`, no added value

</details>

### Key Insights

1. **Pricing relative to market is everything.** `price_diff` alone accounts for ~33% of the production model's predictive power. Hosts who price below the recommended price see dramatically more bookings.

2. **Listing quality matters.** Description length (~38%) and number of photos (~16%) together account for over half of the importance. Turo could nudge hosts to write longer descriptions and upload more photos.

3. **Parking and technology are noise.** Together ~13% of importance. Not decision drivers.

---

## Latency Benchmark

Benchmarked over 1,000 iterations on the containerized Ray Serve deployment (Docker, Apple M-series, single replica).

| Path | Avg | p50 | p95 | p99 |
|------|-----|-----|-----|-----|
| **Raw features** (`POST /predict`) | 13.87 ms | 13.24 ms | 14.38 ms | 25.23 ms |
| **Online store** (`POST /predict/id`) | 13.86 ms | 13.22 ms | 14.62 ms | 24.93 ms |

Both paths have similar latency because **model inference dominates** (~13.7 ms for 100 trees). Feature computation (~0.2 ms) and Redis lookup (~0.17 ms) are negligible. The online store path shows tighter p95/p99, meaning more consistent response times.

<details>
<summary><b>What would change at scale</b></summary>

| Optimization | Impact |
|--------------|--------|
| ONNX export of the RandomForest | 5-10x faster inference (eliminate Python overhead) |
| Batch requests to the Predictor | Amortize Ray serialization across N predictions |
| Multiple Predictor replicas | Linear throughput scaling |
| GPU-based model (XGBoost, neural net) | Leverage Ray Serve's GPU-aware scheduling |
| More complex features (aggregations, embeddings) | Online store path becomes significantly faster than on-the-fly |

</details>

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| **UI** | [localhost:3000](http://localhost:3000) | Next.js frontend |
| **Ray Serve API** | [localhost:8000](http://localhost:8000) | Prediction API |
| **API Docs** | [localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger UI |
| **Airflow** | [localhost:8080](http://localhost:8080) | Pipeline orchestration |
| **MLflow** | [localhost:5001](http://localhost:5001) | Experiment tracking + model registry |
| **Redis** | localhost:6379 | Online feature store + pub/sub |
| **Redis Insight** | [localhost:5540](http://localhost:5540) | Redis GUI |
| **Ray Dashboard** | [localhost:8265](http://localhost:8265) | Ray cluster monitoring |
| **Docs** | [localhost:8100](http://localhost:8100) | MkDocs documentation |
| **Jupyter** | localhost:8888 | EDA notebooks |

## Available Tasks

| Command | Description |
|---|---|
| `mise run setup` | Bootstrap all sub-project deps + pre-commit hooks |
| `mise run dev` | Start all services (mprocs: Docker, UI, Jupyter, Docs) |
| `mise run check` | Full CI check: lint + type check + tests (pre-commit) |
| `mise run seed` | Seed DB + materialize features (local, no Airflow) |
| `mise run materialize` | Compute and materialize features to Parquet + Redis |
| `mise run train` | Train a model locally |
| `mise run promote` | Run champion/challenger promotion locally |
| `mise run pipeline` | Full ML pipeline: seed -> materialize -> train -> promote |

---

## Project Structure

```
training/          ML training pipeline (pandas, sklearn, mlflow)
promotion/         Champion/challenger model promotion (mlflow, redis)
serving/           Ray Serve prediction API (ray, fastapi, feast)
features/          Feast feature store + materialization pipeline
exploration/       EDA notebook (jupytext)
ui/                Next.js + shadcn/ui frontend
airflow/           Airflow DAGs + Dockerfile
data/              Raw CSV datasets (vehicles + reservations)
```

Each sub-project is fully independent with its own `pyproject.toml`, `uv.lock`,
and `.venv`. See each sub-project's README for details.

---

## Design Decisions

<details>
<summary><b>RandomForest over XGBoost/LightGBM</b></summary>

The dataset is small (~500 vehicles) and the feature set simple. RF achieves CV MAE 3.44 with zero tuning; gradient boosting adds complexity and overfitting risk for marginal gain. Easy to swap later -- the pipeline is model-agnostic.
</details>

<details>
<summary><b>Ray Serve over plain FastAPI/Gunicorn</b></summary>

Turo lists Ray as a high-priority technology. Ray Serve's deployment composition lets us scale the Predictor, FeatureLookup, and FeatureMaterializer independently. For a single-model demo this is over-engineered -- but it demonstrates the pattern Turo would use at scale.
</details>

<details>
<summary><b>Feast over a raw Redis client</b></summary>

Feast gives us a unified offline/online store abstraction with point-in-time correctness for training. A raw Redis client would be simpler for serving alone, but wouldn't solve the training-serving skew problem. The tradeoff is operational complexity (registry, materialization) vs. feature consistency guarantees.
</details>

<details>
<summary><b>5 features, not 8</b></summary>

The exploration notebook tested all combinations. `price_diff` alone captures the pricing signal better than separate `actual_price` + `recommended_price` (which are collinear). `price_ratio` is 91% correlated with `price_diff` -- adding it doesn't improve the model but does add a feature to maintain.
</details>

<details>
<summary><b>Parquet offline + Redis online</b></summary>

Training needs all historical vehicles (batch access pattern). Serving needs individual vehicle lookup (point access pattern). Parquet is fast for batch reads; Redis is fast for point lookups. A single store can't serve both patterns well.
</details>

<details>
<summary><b>Separate sub-projects with independent venvs</b></summary>

Each pipeline stage (features, training, promotion, serving) has its own `pyproject.toml` and `.venv`. This mirrors how Turo's MLOps team would structure production code: Airflow shouldn't need scikit-learn, and the serving container shouldn't ship pandas. The tradeoff is duplicated `FEATURE_COLS` definitions across projects -- acceptable for isolation.
</details>

<details>
<summary><b>Champion/challenger promotion over auto-deploy</b></summary>

Every model version gets the `candidate` alias first and must beat the current `champion` on CV MAE before being promoted. This prevents regressions from reaching production. The tradeoff is latency (one extra step) vs. safety.
</details>

<details>
<summary><b>Redis pub/sub for model reload</b></summary>

When a new champion is promoted, the serving layer needs to know. Polling MLflow works but adds latency. Redis pub/sub gives instant notification with zero-downtime reload. The serving layer already depends on Redis (Feast online store), so this adds no new infrastructure.
</details>

---

## Production Considerations

This is a demo -- here's what would change for production:

- **Security** -- CORS is `allow_origins=["*"]` and Airflow uses hardcoded `admin`/`admin`. In production: restrict CORS origins, use a secrets manager, add authentication middleware.

- **Data quality & drift** -- Pydantic validates input schemas, but there's no monitoring for feature distribution drift. In production: add Evidently or Great Expectations to detect training-serving skew.

- **Retraining triggers** -- The pipeline runs on manual trigger. In production, also trigger on data drift detection, model performance degradation, or significant new vehicle volume.

- **Observability** -- Dev includes Ray Dashboard, MLflow UI, and Redis Insight. In production: add structured logging (JSON), Prometheus metrics, and alerting on model staleness or serving errors.

---

## Tech Stack

| Technology | Role |
|------------|------|
| Python 3.12 | Primary language |
| Ray Serve | Model serving (autoscaling, deployment composition) |
| FastAPI | HTTP API (Ray Serve ingress) |
| MLflow | Experiment tracking, model registry |
| Feast | Feature store (offline: Parquet, online: Redis) |
| Airflow | Pipeline orchestration |
| Redis | Online feature store + pub/sub events |
| Docker Compose | Local infrastructure |
| Next.js | Frontend (React, TypeScript, Tailwind) |
| shadcn/ui | UI component library |
| uv | Python package management |
| Ruff + ty | Linting + type checking |

## Dev Tools

```bash
# Setup (one-time)
uvx pre-commit install

# Run all checks
uvx pre-commit run --all-files
```

| Tool | Scope | Purpose |
|------|-------|---------|
| Ruff | Python | Formatting + linting |
| ty | Python (per sub-project) | Type checking |
| ESLint | TypeScript | Linting |
| tsc | TypeScript | Type checking |
| pytest | Python (per sub-project) | Tests |

---

## Agent-Enabled Repository

This repo ships with built-in AI agent support. When you clone it, the
following files give any compatible coding agent (OpenCode, Cursor, Copilot
Workspace, etc.) full project context out of the box:

| File / Directory | Purpose |
|---|---|
| `AGENTS.md` | Always-on context -- Turo role, tech stack, design principles, task specification |
| `.opencode/skills/` | On-demand skills loaded when a task matches their domain |

<details>
<summary><b>Available skills</b></summary>

| Skill | Description |
|---|---|
| **training** | ML training pipeline -- scikit-learn, MLflow tracking, offline store |
| **promotion** | Champion/challenger promotion -- MLflow aliases, Redis pub/sub |
| **serving** | Ray Serve prediction API -- FastAPI, Feast, deployment composition |
| **features** | Feast feature store -- offline/online stores, materialization pipeline |
| **airflow** | Pipeline orchestration -- DAG definitions, BashOperator + uv isolation |
| **ui** | Next.js + shadcn/ui frontend -- Turo design, SSE streams |
| **exploration** | EDA notebook -- Jupytext, data analysis |
| **local-dev** | Local development -- Docker services, mprocs, pipeline triggers, ports |
| **docs** | MkDocs Material documentation site -- structure, conventions |

</details>

No extra setup is needed -- just open the repo with an agent-enabled editor and
the context is picked up automatically.
