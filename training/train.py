"""
Vroom Forecast — Training Pipeline

Trains a Random Forest model to predict the number of reservations per vehicle.
Features are read from the offline feature store (Parquet) via --feature-store.

Usage:
    uv run python -m training --feature-store /feast-data/vehicle_features.parquet --mlflow-uri URI
"""

import argparse
import logging

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "technology",
    "num_images",
    "street_parked",
    "description",
    "price_diff",
]

TARGET_COL = "num_reservations"

MODEL_NAME = "vroom-forecast"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_from_feature_store(parquet_path: str) -> pd.DataFrame:
    """Load pre-computed features from the offline feature store (Parquet).

    The Parquet file is written by the feature materialization pipeline and
    contains all raw + derived features and the label.
    """
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from feature store (%s).", len(df), parquet_path)
    return df


# ── Training ─────────────────────────────────────────────────────────────────


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cv_folds: int = 5,
) -> tuple[RandomForestRegressor, dict]:
    """Train a Random Forest, run cross-validation, and return the model + metrics."""
    model = RandomForestRegressor(**params)

    logger.info("Running %d-fold cross-validation...", cv_folds)
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    logger.info("CV MAE: %.3f (+/- %.3f)", cv_mae, cv_std)

    # Log per-fold scores
    for i, score in enumerate(cv_scores):
        logger.info("  Fold %d MAE: %.3f", i + 1, -score)

    logger.info("Fitting on full dataset...")
    model.fit(X, y)
    preds = model.predict(X)

    metrics = {
        "cv_mae_mean": cv_mae,
        "cv_mae_std": cv_std,
        "train_mae": mean_absolute_error(y, preds),
        "train_rmse": root_mean_squared_error(y, preds),
        "train_r2": r2_score(y, preds),
    }

    # Per-fold CV metrics
    for i, score in enumerate(cv_scores):
        metrics[f"cv_mae_fold_{i + 1}"] = float(-score)

    for feat, imp in zip(X.columns, model.feature_importances_, strict=True):
        metrics[f"importance_{feat}"] = imp

    return model, metrics


# ── Model registration ───────────────────────────────────────────────────────


def register_model(run_id: str, model_name: str) -> str:
    """Register a new model version in MLflow, tag it as 'candidate', and return its version."""
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException as exc:
        if getattr(exc, "error_code", "") != "RESOURCE_DOES_NOT_EXIST":
            raise
        logger.info("Registered model '%s' not found, creating it.", model_name)
        client.create_registered_model(model_name)

    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    logger.info("Registered model version %s for '%s'.", mv.version, model_name)

    client.set_registered_model_alias(model_name, "candidate", mv.version)
    logger.info("Alias 'candidate' set on version %s.", mv.version)

    return mv.version


# ── Pipeline ─────────────────────────────────────────────────────────────────


def run(
    mlflow_uri: str,
    experiment_name: str,
    model_name: str,
    feature_store: str,
) -> str:
    """Full training pipeline. Returns the registered model version string."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", mlflow_uri)
    logger.info("MLflow experiment: %s", experiment_name)

    df = load_from_feature_store(feature_store)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    with mlflow.start_run(run_name="random_forest_pipeline"):
        mlflow.log_params(RF_PARAMS)
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("pipeline", "train.py")
        mlflow.set_tag("feature_source", "feature_store")

        model, metrics = train_and_evaluate(X, y, RF_PARAMS)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X.head(1))

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run")
        run_id = active_run.info.run_id
        logger.info("Run ID: %s", run_id)

    version = register_model(run_id, model_name)
    logger.info(
        "Done. Model registered as %s v%s (alias: candidate).",
        model_name,
        version,
    )
    return version


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Vroom Forecast training pipeline")
    parser.add_argument(
        "--feature-store",
        type=str,
        required=True,
        help="Path to the offline feature store Parquet file",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5001",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="vroom-forecast",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Registered model name in MLflow",
    )
    return parser.parse_args()
