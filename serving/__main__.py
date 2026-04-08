"""Run the serving API via Ray Serve: python -m serving."""

import time

import ray
from ray import serve

from serving.app import VroomForecastApp
from serving.config import settings
from serving.model import (
    FeatureComputer,
    FeatureLookup,
    FeatureMaterializer,
    ModelReloadListener,
    OfflineFeatureReader,
    Predictor,
)


def main() -> None:
    """Initialize Ray, compose Serve deployments, and start background actors."""
    ray.init(
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
    )

    # Configure HTTP
    serve.start(http_options={"host": settings.host, "port": settings.port})

    # Compose Serve deployments
    predictor = Predictor.bind()
    feature_computer = FeatureComputer.bind()
    feature_lookup = FeatureLookup.bind()
    offline_reader = OfflineFeatureReader.bind()
    ingress = VroomForecastApp.bind(predictor, feature_computer, feature_lookup, offline_reader)

    serve.run(ingress, name="vroom-forecast")

    # Get a handle to the Predictor for the reload listener.
    # serve.get_deployment_handle() returns a handle to a specific deployment,
    # not the ingress — this is needed for the ModelReloadListener to call
    # Predictor.reload() directly.
    predictor_handle = serve.get_deployment_handle("Predictor", "vroom-forecast")

    # Start background Ray actors (not Serve deployments — they don't handle HTTP)
    # 1. Feature materializer: subscribes to vehicle-saved events, writes to Feast
    materializer = FeatureMaterializer.remote()
    materializer.run.remote()

    # 2. Model reload listener: subscribes to model-promoted events, reloads Predictor
    reload_listener = ModelReloadListener.remote(predictor_handle)
    reload_listener.run.remote()

    print(f"Ray Serve running at http://{settings.host}:{settings.port}")
    print(f"Ray dashboard at http://{settings.host}:8265")

    # Block until interrupted (cross-platform)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
