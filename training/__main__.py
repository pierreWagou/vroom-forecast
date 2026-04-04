"""Run the training pipeline: python -m training [OPTIONS]."""

from training.train import parse_args, run

if __name__ == "__main__":
    args = parse_args()
    version = run(
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        model_name=args.model_name,
        feature_store=args.feature_store,
        data_dir=args.data_dir,
    )
    print(version)
