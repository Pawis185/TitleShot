"""Command line helpers for running the perception pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .model import PerceptionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TitleShot perception pipeline")
    parser.add_argument("image", type=Path, help="Path to the product image")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional textual hint (title, bullet point, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the JSON output",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed_args = args or parse_args()
    pipeline = PerceptionPipeline()
    result = pipeline.run(parsed_args.image, text_hint=parsed_args.text)
    payload = result.to_dict()
    json_payload = json.dumps(payload, indent=2)
    if parsed_args.output:
        parsed_args.output.write_text(json_payload, encoding="utf-8")
    else:
        print(json_payload)


if __name__ == "__main__":
    main()