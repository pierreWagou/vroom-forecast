"""Run the promotion pipeline: python -m promotion [OPTIONS]."""

from promotion.promote import parse_args, promote

if __name__ == "__main__":
    args = parse_args()
    promoted = promote(
        mlflow_uri=args.mlflow_uri,
        model_name=args.model_name,
        candidate_version=args.version,
        candidate_alias=args.candidate_alias,
        metric_name=args.metric,
    )
    print("promoted" if promoted else "retained")
