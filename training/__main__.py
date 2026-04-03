"""Run the training pipeline: python -m training [OPTIONS]."""

from training.train import parse_args, run

if __name__ == "__main__":
    args = parse_args()
    version = run(
        data_dir=args.data_dir,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        model_name=args.model_name,
    )
    print(version)
