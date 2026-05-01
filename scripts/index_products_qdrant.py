import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path so `service.*` imports work
# when this script is run directly from terminal.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from service.semantic_search import upsert_products_to_qdrant


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy products from PostgreSQL into Qdrant with embeddings "
            "so semantic search can work."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="How many products to index at most (default: 2000).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete old Qdrant collection and create a fresh one before indexing. "
            "Use this when you want a full rebuild."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run indexing and print a beginner-friendly summary."""
    options = parse_args()

    print("Starting product indexing...")
    print(f"- limit: {options.limit}")
    print(f"- recreate collection: {options.recreate}")

    result = upsert_products_to_qdrant(
        limit=options.limit,
        recreate=options.recreate,
    )

    print("\nDone. Indexing result:")
    print(json.dumps(result, indent=2))
    print("\nTip: You can now ask semantic queries like:")
    print("  'show aesthetic shirts for summer'")


if __name__ == "__main__":
    main()
