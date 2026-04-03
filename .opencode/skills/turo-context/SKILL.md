---
name: turo-context
description: Turo Staff MLOps Engineer role — tech stack priorities, design principles, and what Staff-level judgment means for architecture decisions
---

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

## MLflow Model Lifecycle

- Training registers a new version and sets `candidate` alias
- Promotion compares `candidate` vs `champion` by `cv_mae_mean` (lower is better)
- If candidate is better, `champion` alias moves to the new version

## What Staff-Level Means Here

- Pragmatic architecture with clear reasoning for tradeoffs
- Awareness of what changes at scale (even if the demo is small)
- Clean, well-documented code a team can maintain
- Honest about limitations (e.g., "this isn't truly event-driven, here's why")
- Show the reviewer you understand the full ML lifecycle, not just one piece
