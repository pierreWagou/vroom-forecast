"""Tests for the promotion pipeline — uses mocked MLflow client."""

from unittest.mock import MagicMock, patch

from promote import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    resolve_candidate_version,
)

# ── resolve_candidate_version ────────────────────────────────────────────────


class TestResolveCandidateVersion:
    def test_explicit_version_returned_directly(self) -> None:
        client = MagicMock()
        result = resolve_candidate_version(client, "model", "5", CANDIDATE_ALIAS)
        assert result == "5"
        client.get_model_version_by_alias.assert_not_called()

    def test_alias_resolved_when_no_version(self) -> None:
        client = MagicMock()
        mv = MagicMock()
        mv.version = "3"
        client.get_model_version_by_alias.return_value = mv

        result = resolve_candidate_version(client, "model", None, "candidate")
        assert result == "3"
        client.get_model_version_by_alias.assert_called_once_with("model", "candidate")


# ── promote (integration with mocked MLflow) ─────────────────────────────────


class TestPromote:
    def _make_run(self, metric_name: str, metric_value: float) -> MagicMock:
        run = MagicMock()
        run.data.metrics = {metric_name: metric_value}
        return run

    def _make_mv(self, version: str, run_id: str = "run-123") -> MagicMock:
        mv = MagicMock()
        mv.version = version
        mv.run_id = run_id
        return mv

    @patch("promote.mlflow")
    def test_first_champion_promoted(self, mock_mlflow: MagicMock) -> None:
        from mlflow.exceptions import MlflowException
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        # Candidate exists
        client.get_model_version.return_value = self._make_mv("1")
        client.get_run.return_value = self._make_run("cv_mae_mean", 2.5)

        # No champion
        client.get_model_version_by_alias.side_effect = MlflowException("not found")

        result = promote(
            mlflow_uri="http://fake",
            model_name="model",
            metric_name="cv_mae_mean",
            candidate_version="1",
        )

        assert result is True
        client.set_registered_model_alias.assert_called_once_with("model", CHAMPION_ALIAS, "1")

    @patch("promote.mlflow")
    def test_better_candidate_promoted(self, mock_mlflow: MagicMock) -> None:
        from mlflow.exceptions import MlflowException
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        # Candidate: MAE = 2.0
        client.get_model_version.return_value = self._make_mv("2", "run-candidate")
        # Champion: MAE = 3.0
        champion_mv = self._make_mv("1", "run-champion")

        def get_run_side_effect(run_id: str) -> MagicMock:
            if run_id == "run-candidate":
                return self._make_run("cv_mae_mean", 2.0)
            return self._make_run("cv_mae_mean", 3.0)

        client.get_run.side_effect = get_run_side_effect
        client.get_model_version_by_alias.return_value = champion_mv

        result = promote(
            mlflow_uri="http://fake",
            model_name="model",
            metric_name="cv_mae_mean",
            candidate_version="2",
        )

        assert result is True
        client.set_registered_model_alias.assert_called_once_with("model", CHAMPION_ALIAS, "2")

    @patch("promote.mlflow")
    def test_worse_candidate_retained(self, mock_mlflow: MagicMock) -> None:
        from mlflow.exceptions import MlflowException
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        # Candidate: MAE = 4.0 (worse)
        client.get_model_version.return_value = self._make_mv("2", "run-candidate")
        # Champion: MAE = 3.0
        champion_mv = self._make_mv("1", "run-champion")

        def get_run_side_effect(run_id: str) -> MagicMock:
            if run_id == "run-candidate":
                return self._make_run("cv_mae_mean", 4.0)
            return self._make_run("cv_mae_mean", 3.0)

        client.get_run.side_effect = get_run_side_effect
        client.get_model_version_by_alias.return_value = champion_mv

        result = promote(
            mlflow_uri="http://fake",
            model_name="model",
            metric_name="cv_mae_mean",
            candidate_version="2",
        )

        assert result is False
        client.set_registered_model_alias.assert_not_called()

    @patch("promote.mlflow")
    def test_equal_candidate_retained(self, mock_mlflow: MagicMock) -> None:
        from mlflow.exceptions import MlflowException
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        # Same score
        client.get_model_version.return_value = self._make_mv("2", "run-candidate")
        champion_mv = self._make_mv("1", "run-champion")

        def get_run_side_effect(run_id: str) -> MagicMock:
            return self._make_run("cv_mae_mean", 3.0)

        client.get_run.side_effect = get_run_side_effect
        client.get_model_version_by_alias.return_value = champion_mv

        result = promote(
            mlflow_uri="http://fake",
            model_name="model",
            metric_name="cv_mae_mean",
            candidate_version="2",
        )

        assert result is False

    @patch("promote.mlflow")
    def test_missing_metric_returns_false(self, mock_mlflow: MagicMock) -> None:
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client

        # Candidate has no metrics
        client.get_model_version.return_value = self._make_mv("1")
        run = MagicMock()
        run.data.metrics = {}
        client.get_run.return_value = run

        result = promote(
            mlflow_uri="http://fake",
            model_name="model",
            metric_name="cv_mae_mean",
            candidate_version="1",
        )

        assert result is False
