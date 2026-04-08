"""Tests for the promotion pipeline — uses mocked MLflow client."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from promote import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    REDIS_CHANNEL,
    _notify_promoted,
    parse_args,
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

        # No champion — simulate the real MlflowException with error_code attribute
        no_champion_exc = MlflowException("Registered model alias 'champion' not found.")
        no_champion_exc.error_code = "RESOURCE_DOES_NOT_EXIST"
        client.get_model_version_by_alias.side_effect = no_champion_exc

        result = promote(
            mlflow_uri="http://fake",
            model_name="model",
            metric_name="cv_mae_mean",
            candidate_version="1",
        )

        assert result is True
        client.set_registered_model_alias.assert_called_once_with("model", CHAMPION_ALIAS, "1")

    @patch("promote.mlflow")
    def test_first_champion_promoted_invalid_parameter(self, mock_mlflow: MagicMock) -> None:
        """MLflow 3.x returns INVALID_PARAMETER_VALUE when alias doesn't exist."""
        from mlflow.exceptions import MlflowException
        from promote import CANDIDATE_ALIAS, promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        client.get_model_version.return_value = self._make_mv("1")
        client.get_run.return_value = self._make_run("cv_mae_mean", 2.5)

        # Candidate alias resolves fine; champion alias raises
        no_champion_exc = MlflowException("Registered model alias champion not found.")
        no_champion_exc.error_code = "INVALID_PARAMETER_VALUE"

        def alias_side_effect(name: str, alias: str) -> MagicMock:
            if alias == CANDIDATE_ALIAS:
                return self._make_mv("1")
            raise no_champion_exc

        client.get_model_version_by_alias.side_effect = alias_side_effect

        result = promote(mlflow_uri="http://test", model_name="model", metric_name="cv_mae_mean")

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

    @patch("promote.mlflow")
    def test_champion_missing_metric_promotes_candidate(self, mock_mlflow: MagicMock) -> None:
        """Champion exists but has no metric — candidate is promoted by default."""
        from mlflow.exceptions import MlflowException
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        # Candidate: has metric
        client.get_model_version.return_value = self._make_mv("2", "run-candidate")

        # Champion: exists but has NO metric
        champion_mv = self._make_mv("1", "run-champion")
        champion_run = MagicMock()
        champion_run.data.metrics = {}

        def get_run_side_effect(run_id: str) -> MagicMock:
            if run_id == "run-candidate":
                return self._make_run("cv_mae_mean", 2.5)
            return champion_run

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


# ── _notify_promoted ─────────────────────────────────────────────────────────


class TestNotifyPromoted:
    def _make_mv(self, version: str, run_id: str = "run-123") -> MagicMock:
        mv = MagicMock()
        mv.version = version
        mv.run_id = run_id
        return mv

    def _make_run(self, metric_name: str, metric_value: float) -> MagicMock:
        r = MagicMock()
        r.data.metrics = {metric_name: metric_value}
        return r

    def test_noop_when_no_redis_url(self) -> None:
        """Should return immediately without error when redis_url is None."""
        _notify_promoted(None, "model", "1")  # should not raise

    def test_publishes_to_correct_channel(self) -> None:
        """Should publish a JSON payload to the correct Redis channel."""
        mock_redis_mod = MagicMock()
        mock_r = MagicMock()
        mock_redis_mod.from_url.return_value = mock_r
        mock_r.publish.return_value = 2

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            _notify_promoted("redis://localhost:6379", "m", "3")

        mock_redis_mod.from_url.assert_called_once_with("redis://localhost:6379")
        mock_r.publish.assert_called_once()
        call_args = mock_r.publish.call_args
        assert call_args[0][0] == REDIS_CHANNEL
        payload = json.loads(call_args[0][1])
        assert payload == {"model_name": "m", "version": "3"}
        mock_r.close.assert_called_once()

    def test_closes_connection_on_publish_failure(self) -> None:
        """Should close the Redis connection even if publish raises."""
        mock_redis_mod = MagicMock()
        mock_r = MagicMock()
        mock_redis_mod.from_url.return_value = mock_r
        mock_r.publish.side_effect = ConnectionError("Connection refused")

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            _notify_promoted("redis://localhost:6379", "m", "1")

        mock_r.close.assert_called_once()

    def test_handles_connection_failure(self) -> None:
        """Should not raise if redis.from_url itself fails."""
        mock_redis_mod = MagicMock()
        mock_redis_mod.from_url.side_effect = ConnectionError("Cannot connect")

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            _notify_promoted("redis://bad-host:6379", "m", "1")  # should not raise

    @patch("promote.mlflow")
    def test_notification_called_on_promotion(self, mock_mlflow: MagicMock) -> None:
        """Integration: promote() calls _notify_promoted when promoting."""
        from mlflow.exceptions import MlflowException
        from promote import promote

        client = MagicMock()
        mock_mlflow.MlflowClient.return_value = client
        mock_mlflow.exceptions.MlflowException = MlflowException

        client.get_model_version.return_value = self._make_mv("1")
        client.get_run.return_value = self._make_run("cv_mae_mean", 2.5)

        no_champion_exc = MlflowException("Not found")
        no_champion_exc.error_code = "RESOURCE_DOES_NOT_EXIST"
        client.get_model_version_by_alias.side_effect = no_champion_exc

        with patch("promote._notify_promoted") as mock_notify:
            result = promote(
                mlflow_uri="http://fake",
                model_name="model",
                metric_name="cv_mae_mean",
                candidate_version="1",
                redis_url="redis://localhost:6379",
            )

        assert result is True
        mock_notify.assert_called_once_with("redis://localhost:6379", "model", "1")


# ── parse_args ───────────────────────────────────────────────────────────────


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["promote"])
        args = parse_args()
        assert args.version is None
        assert args.candidate_alias == CANDIDATE_ALIAS
        assert args.mlflow_uri == "http://localhost:5001"
        assert args.model_name == "vroom-forecast"
        assert args.metric == "cv_mae_mean"
        assert args.redis_url is None

    def test_custom_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "promote",
                "--version",
                "5",
                "--mlflow-uri",
                "http://custom:5000",
                "--model-name",
                "my-model",
                "--metric",
                "custom_metric",
                "--redis-url",
                "redis://myredis:6379",
            ],
        )
        args = parse_args()
        assert args.version == "5"
        assert args.mlflow_uri == "http://custom:5000"
        assert args.model_name == "my-model"
        assert args.metric == "custom_metric"
        assert args.redis_url == "redis://myredis:6379"
