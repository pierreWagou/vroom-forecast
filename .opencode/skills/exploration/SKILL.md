---
name: exploration
description: EDA notebook — Jupytext percent format, data analysis, findings that informed model design
---

## Role

You are a data scientist doing exploratory analysis.

## Overview

Independent uv project (`exploration/`) for exploratory data analysis via a
Jupytext-synced notebook.

## Rules

- This is notebook-style code; relaxed lint rules apply (E402, B905 ignored)
- Not type-checked — excluded from pre-commit ty hooks
- Jupytext percent format; keep the `.py` file as source of truth
- Findings here inform the training pipeline but don't share code directly

## Notebook

`exploration/exploration.py` — Jupytext percent format (`.py` file, syncs to
`.ipynb` automatically). Contains the EDA that produced the key findings in
the README (feature importance, correlation analysis, distribution plots).

## File Layout

```
exploration/
  exploration.py     # Jupytext notebook (percent format)
  pyproject.toml
  uv.lock
```

## Dependencies

pandas, scikit-learn, mlflow, matplotlib, seaborn, ipykernel, jupytext, notebook.

## Run Locally

```bash
cd exploration && uv run jupyter notebook
```

Jupyter UI: http://localhost:8888

## Standards

- Format/lint: ruff (root `ruff.toml`) with relaxed rules: `E402` (late imports)
  and `B905` (zip without strict) are ignored for `exploration/*`
- No type checking (excluded from pre-commit)
