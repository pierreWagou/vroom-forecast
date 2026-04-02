"""
Vroom Forecast — Model Promotion

Compares a candidate model version against the current champion and promotes
it if it has a better cv_mae_mean.

Usage:
    uv run python promote.py --version VERSION [--mlflow-uri URI] [--model-name NAME]
                             [--metric METRIC]
"""

import argparse
import logging
import sys

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "vroom-forecast"
CHAMPION_ALIAS = "champion"
COMPARISON_METRIC = "cv_mae_mean"  # lower is better


def promote(
    mlflow_uri: str,
    model_name: str,
    candidate_version: str,
    metric_name: str,
) -> bool:
    """Compare candidate against champion and promote if better.

    Returns True if the candidate was promoted, False otherwise.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.MlflowClient()

    # Fetch candidate
    candidate_mv = client.get_model_version(model_name, candidate_version)
    candidate_run = client.get_run(candidate_mv.run_id)
    candidate_metric = candidate_run.data.metrics.get(metric_name)

    if candidate_metric is None:
        logger.error(
            "Candidate v%s (run %s) has no metric '%s'. Cannot compare.",
            candidate_version,
            candidate_mv.run_id,
            metric_name,
        )
        return False

    logger.info(
        "Candidate v%s: %s=%.4f",
        candidate_version,
        metric_name,
        candidate_metric,
    )

    # Check if there is a current champion
    try:
        champion_mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        champion_run = client.get_run(champion_mv.run_id)
        champion_metric = champion_run.data.metrics.get(metric_name)

        if champion_metric is None:
            logger.warning(
                "Current champion v%s has no metric '%s'. Promoting candidate by default.",
                champion_mv.version,
                metric_name,
            )
            client.set_registered_model_alias(
                model_name, CHAMPION_ALIAS, candidate_version
            )
            logger.info("Version %s promoted as champion.", candidate_version)
            return True

        logger.info(
            "Current champion v%s: %s=%.4f",
            champion_mv.version,
            metric_name,
            champion_metric,
        )

        if candidate_metric < champion_metric:
            client.set_registered_model_alias(
                model_name, CHAMPION_ALIAS, candidate_version
            )
            logger.info(
                "New champion! Version %s promoted (%.4f < %.4f).",
                candidate_version,
                candidate_metric,
                champion_metric,
            )
            return True
        else:
            logger.info(
                "Current champion v%s retained (candidate %.4f >= champion %.4f).",
                champion_mv.version,
                candidate_metric,
                champion_metric,
            )
            return False

    except mlflow.exceptions.MlflowException:
        # No champion yet
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, candidate_version)
        logger.info(
            "No existing champion. Version %s promoted as first champion.",
            candidate_version,
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model version to champion")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version to promote (e.g. '3')",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5001",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Registered model name in MLflow",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=COMPARISON_METRIC,
        help="Metric to compare (lower is better)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    promoted = promote(
        mlflow_uri=args.mlflow_uri,
        model_name=args.model_name,
        candidate_version=args.version,
        metric_name=args.metric,
    )
    sys.exit(0 if promoted else 1)
