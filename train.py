"""
Vroom Forecast — Training Pipeline

Loads vehicle and reservation data, engineers features, trains a Random Forest
model to predict the number of reservations per vehicle, evaluates with
cross-validation, logs everything to MLflow, and registers the model.

Usage:
    uv run python train.py [--data-dir DATA_DIR] [--mlflow-uri URI] [--experiment NAME]
                           [--model-name NAME]
"""

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "technology",
    "actual_price",
    "recommended_price",
    "num_images",
    "street_parked",
    "description",
    "price_diff",
    "price_ratio",
]

TARGET_COL = "num_reservations"

MODEL_NAME = "vroom-forecast"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load vehicles and reservations, aggregate, and merge."""
    vehicles_path = data_dir / "vehicles.csv"
    reservations_path = data_dir / "reservations.csv"

    logger.info("Loading vehicles from %s", vehicles_path)
    vehicles = pd.read_csv(vehicles_path)
    logger.info("Vehicles shape: %s", vehicles.shape)

    logger.info("Loading reservations from %s", reservations_path)
    reservations = pd.read_csv(reservations_path, parse_dates=["created_at"])
    logger.info("Reservations shape: %s", reservations.shape)

    res_counts = reservations.groupby("vehicle_id").size().reset_index(name=TARGET_COL)

    df = vehicles.merge(res_counts, on="vehicle_id", how="left")
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    logger.info(
        "Merged dataset: %d vehicles, %d with 0 reservations",
        len(df),
        (df[TARGET_COL] == 0).sum(),
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived pricing features."""
    df = df.copy()
    df["price_diff"] = df["actual_price"] - df["recommended_price"]
    df["price_ratio"] = df["actual_price"] / df["recommended_price"]
    return df


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cv_folds: int = 5,
) -> tuple[RandomForestRegressor, dict]:
    """Train a Random Forest, run cross-validation, and return the model + metrics."""
    model = RandomForestRegressor(**params)

    logger.info("Running %d-fold cross-validation...", cv_folds)
    cv_scores = cross_val_score(
        model, X, y, cv=cv_folds, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    logger.info("CV MAE: %.3f (+/- %.3f)", cv_mae, cv_std)

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

    # Per-feature importances
    for feat, imp in zip(X.columns, model.feature_importances_):
        metrics[f"importance_{feat}"] = imp

    return model, metrics


def register_model(run_id: str, model_name: str) -> str:
    """Register a new model version in MLflow and return its version number."""
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Ensure the registered model exists
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        logger.info("Registered model '%s' not found, creating it.", model_name)
        client.create_registered_model(model_name)

    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    logger.info("Registered model version %s for '%s'.", mv.version, model_name)
    return mv.version


def run(data_dir: Path, mlflow_uri: str, experiment_name: str, model_name: str) -> None:
    """Full training pipeline."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", mlflow_uri)
    logger.info("MLflow experiment: %s", experiment_name)

    df = load_data(data_dir)
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    with mlflow.start_run(run_name="random_forest_pipeline"):
        mlflow.log_params(RF_PARAMS)
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("pipeline", "train.py")

        model, metrics = train_and_evaluate(X, y, RF_PARAMS)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X.head(1))

        run_id = mlflow.active_run().info.run_id
        logger.info("Run ID: %s", run_id)

    version = register_model(run_id, model_name)
    logger.info(
        "Done. Model registered as %s v%s. "
        "Run: uv run python promote.py --version %s  to promote it.",
        model_name,
        version,
        version,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vroom Forecast training pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing vehicles.csv and reservations.csv",
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


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        model_name=args.model_name,
    )
