"""Run the promotion pipeline: python -m promotion [OPTIONS]."""

import logging
import sys

from promotion.promote import parse_args, promote

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parse_args()
    try:
        promoted = promote(
            mlflow_uri=args.mlflow_uri,
            model_name=args.model_name,
            candidate_version=args.version,
            candidate_alias=args.candidate_alias,
            metric_name=args.metric,
            redis_url=args.redis_url,
        )
        # Exit 0 for both promoted and retained — these are valid outcomes
        print("promoted" if promoted else "retained")
    except Exception:
        logger.exception("Promotion failed.")
        print("error", file=sys.stderr)
        sys.exit(1)
