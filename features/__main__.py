"""Run the feature pipeline: python -m features."""

from pipeline import parse_args, run

if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        feast_repo=args.feast_repo,
        parquet_path=args.parquet_path,
        vehicles_db=args.vehicles_db,
    )
