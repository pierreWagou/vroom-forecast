---
mode: agent
description: Vroom forecast task — vehicle reservation analysis, model training, and serving pipeline
---

# Vroom Forecast Task

## Datasets

Two datasets are available in `data/`:

**`vehicles.csv`** — all vehicles and their attributes:
- `technology`: 0 = none, 1 = installed (makes vehicle "instantly bookable" and unlockable via mobile)
- `actual_price`: daily price set by the owner
- `recommended_price`: market price based on internal analysis
- `num_images`: number of photos uploaded by the owner
- `street_parking`: 0 = no, 1 = yes
- `description`: number of characters in the owner's description

**`reservations.csv`** — all completed reservations:
- `vehicle_id`: vehicle's unique ID
- `created_at`: timestamp when the reservation was created

## Tasks

1. **Which factors drive total # of reservations?**
   - Aggregate reservations per vehicle, join with vehicle attributes
   - Train a model to identify the most important features
   - Present key insights

2. **End-to-end pipeline from training to serving**
   - Train the best model found above
   - Build a containerized FastAPI app that serves predictions
   - The service must be testable via a locally hosted API call
   - Benchmark and report the average latency of the service
