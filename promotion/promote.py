"""
Vroom Forecast — Model Promotion

Compares a candidate model version against the current champion and promotes
it if it has a better cv_mae_mean. The candidate can be specified by explicit
version number or by MLflow alias (default: "candidate").

Usage:
    uv run python -m promotion [--version VERSION | --candidate-alias ALIAS]
                               [--mlflow-uri URI] [--model-name NAME]
                               [--metric METRIC]
"""

import argparse
import logging

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "vroom-forecast"
CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"
COMPARISON_METRIC = "cv_mae_mean"  # lower is better


def resolve_candidate_version(
    client: mlflow.MlflowClient,
    model_name: str,
    version: str | None,
    alias: str,
) -> str:
    """Return the candidate model version, either from an explicit arg or an alias."""
    if version:
        return version
    mv = client.get_model_version_by_alias(model_name, alias)
    logger.info("Resolved alias '%s' to version %s.", alias, mv.version)
    return mv.version


def promote(
    mlflow_uri: str,
    model_name: str,
    metric_name: str,
    candidate_version: str | None = None,
    candidate_alias: str = CANDIDATE_ALIAS,
) -> bool:
    """Compare candidate against champion and promote if better.

    Returns True if the candidate was promoted, False otherwise.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.MlflowClient()

    # Resolve candidate
    version = resolve_candidate_version(client, model_name, candidate_version, candidate_alias)

    # Fetch candidate metrics
    candidate_mv = client.get_model_version(model_name, version)
    assert candidate_mv.run_id is not None, f"Model version {version} has no associated run"
    candidate_run = client.get_run(candidate_mv.run_id)
    candidate_metric = candidate_run.data.metrics.get(metric_name)

    if candidate_metric is None:
        logger.error(
            "Candidate v%s (run %s) has no metric '%s'. Cannot compare.",
            version,
            candidate_mv.run_id,
            metric_name,
        )
        return False

    logger.info(
        "Candidate v%s: %s=%.4f",
        version,
        metric_name,
        candidate_metric,
    )

    # Check if there is a current champion
    try:
        champion_mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        assert champion_mv.run_id is not None, "Champion version has no associated run"
        champion_run = client.get_run(champion_mv.run_id)
        champion_metric = champion_run.data.metrics.get(metric_name)

        if champion_metric is None:
            logger.warning(
                "Current champion v%s has no metric '%s'. Promoting candidate by default.",
                champion_mv.version,
                metric_name,
            )
            client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
            logger.info("Version %s promoted as champion.", version)
            return True

        logger.info(
            "Current champion v%s: %s=%.4f",
            champion_mv.version,
            metric_name,
            champion_metric,
        )

        if candidate_metric < champion_metric:
            client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
            logger.info(
                "New champion! Version %s promoted (%.4f < %.4f).",
                version,
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
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
        logger.info(
            "No existing champion. Version %s promoted as first champion.",
            version,
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model version to champion")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version to evaluate (e.g. '3'). If omitted, uses --candidate-alias.",
    )
    parser.add_argument(
        "--candidate-alias",
        type=str,
        default=CANDIDATE_ALIAS,
        help="MLflow alias to resolve the candidate version (default: 'candidate')",
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
