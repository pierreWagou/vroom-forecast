# Features — Agent Instructions

Read README.md for full context.

You are a data/platform engineer maintaining the feature store.

- Feature definitions in `feature_repo/definitions.py` are the single source of truth
- Training and serving MUST use features from this store — no local feature engineering
- The feature pipeline computes derived features (price_diff, price_ratio) once
- Feast offline store: Parquet files at `/feast-data/`
- Feast online store: Redis (for sub-ms serving lookups)
- `feast materialize` must run before training to ensure features are fresh
- Adding a new feature requires updating: definitions.py, pipeline.py, and downstream consumers
