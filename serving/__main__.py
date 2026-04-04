"""Run the serving API via Ray Serve: python -m serving."""

import ray
from ray import serve

from serving.app import VroomForecastApp
from serving.config import settings
from serving.model import FeatureComputer, FeatureLookup, FeatureMaterializer, Predictor


def main() -> None:
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
    ingress = VroomForecastApp.bind(predictor, feature_computer, feature_lookup)

    serve.run(ingress, name="vroom-forecast")

    # Start the background feature materializer actor
    # This is a Ray actor (not a Serve deployment) — it subscribes to Redis
    # pub/sub and writes to the Feast online store when vehicles are saved.
    materializer = FeatureMaterializer.remote()
    materializer.run.remote()  # fire-and-forget: runs forever in the background

    print(f"Ray Serve running at http://{settings.host}:{settings.port}")
    print(f"Ray dashboard at http://{settings.host}:8265")

    try:
        import signal

        signal.pause()
    except KeyboardInterrupt:
        print("Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
