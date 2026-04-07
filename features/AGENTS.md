# Features — Agent Instructions

Read README.md for full context.

You are a data/platform engineer maintaining the feature store.

- Feature definitions in `feature_repo/definitions.py` are the single source of truth
- Training and serving MUST use features from this store — no local feature engineering
- The feature pipeline computes derived features (price_diff, price_ratio) once
- Feast offline store: Parquet files at `/feast-data/` — contains ALL vehicles
- Feast online store: Redis — contains only new arrivals (num_reservations IS NULL)
- The batch pipeline writes to Parquet (all) and Redis (new arrivals only via write_to_online_store)
- The real-time FeatureMaterializer writes individual vehicles to Redis on save
- `num_reservations` is nullable: NULL for new arrivals (no observation), int for fleet (observed count, including 0)
- Adding a new feature requires updating: definitions.py, pipeline.py, and downstream consumers
