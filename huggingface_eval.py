"""Run lightweight evaluations against Hugging Face Inference API.

This script reads the image URLs from ``dataset.json`` and queries a set of
multi-modal models hosted on Hugging Face via the OpenAI-compatible
``chat/completions`` endpoint. Each response is expected to return JSON with
basic e-commerce attributes so we can measure simple category accuracy and
record latency.

Usage example:

```
python huggingface_eval.py \
    --dataset dataset.json \
    --models qwen2-vl-7b internvl2-8b llava-1.5-7b \
    --output hf_results.json
```

Environment variables:
    * ``HF_API_TOKEN``: Optional token for authenticated Inference API access.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from huggingface_hub import InferenceAPIError, InferenceClient


DEFAULT_MODELS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
}


def parse_args() -> argparse.Namespace:
    """CLI argument parsing."""

    parser = argparse.ArgumentParser(
        description="Evaluate perception models via Hugging Face Inference API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset.json"),
        help="Path to dataset JSON file containing image URLs and categories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS.keys()),
        help="List of model aliases to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hf_eval_results.json"),
        help="Where to write aggregated evaluation results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of images evaluated (useful for quick smoke tests)",
    )

    return parser.parse_args()


def load_dataset(dataset_path: Path, max_samples: Optional[int]) -> List[Dict[str, str]]:
    """Flatten the dataset.json structure into a simple list of records."""

    data: Dict[str, List[Dict[str, str]]] = json.loads(dataset_path.read_text())
    records: List[Dict[str, str]] = []

    for category, items in data.items():
        for item in items:
            records.append({"category": category, "image_url": item["image_url"]})

    if max_samples is not None:
        return records[:max_samples]
    return records


def _extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON block from model output.

    Some models may return text before/after JSON, so we search for the first
    JSON object and attempt to parse it.
    """

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"{.*}", content, flags=re.S)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def _coerce_content_to_text(content: Any) -> str:
    """Normalize Hugging Face chat completion content to a plain string."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "".join(text_parts)

    return str(content)


def _build_prompt(expected_category: str) -> str:
    return (
        "You are an expert e-commerce analyst. Identify the product attributes "
        "from the image and respond ONLY with JSON using the keys: category, "
        "color, material, style, selling_points. Category should be one of the "
        "dataset labels (e.g., Electronics, Clothing, Furniture, Toys, Beauty). "
        f"The expected category for this example is '{expected_category}'."
    )


def evaluate_model(
    model_alias: str,
    model_id: str,
    records: Iterable[Dict[str, str]],
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single model and return aggregate metrics and raw outputs."""

    client = InferenceClient(model=model_id, token=token)

    results: List[Dict[str, Any]] = []
    successes = 0
    category_matches = 0

    for record in records:
        prompt = _build_prompt(record["category"])
        start = time.perf_counter()

        try:
            completion = client.chat_completion(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": record["image_url"]}},
                        ],
                    }
                ],
                max_tokens=500,
            )
            latency = time.perf_counter() - start

            content = _coerce_content_to_text(completion.choices[0].message["content"])
            parsed = _extract_json_from_response(content)

            is_success = parsed is not None
            predicted_category = (parsed or {}).get("category", "")
            matches_category = bool(
                predicted_category
                and record["category"].lower() in predicted_category.lower()
            )

            successes += int(is_success)
            category_matches += int(matches_category)

            results.append(
                {
                    "image_url": record["image_url"],
                    "expected_category": record["category"],
                    "raw_response": content,
                    "parsed": parsed,
                    "latency_sec": latency,
                    "parsed_ok": is_success,
                    "category_match": matches_category,
                }
            )

        except InferenceAPIError as err:
            results.append(
                {
                    "image_url": record["image_url"],
                    "expected_category": record["category"],
                    "error": str(err),
                    "latency_sec": None,
                    "parsed_ok": False,
                    "category_match": False,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            results.append(
                {
                    "image_url": record["image_url"],
                    "expected_category": record["category"],
                    "error": str(exc),
                    "latency_sec": None,
                    "parsed_ok": False,
                    "category_match": False,
                }
            )

    total = len(results)
    return {
        "model_alias": model_alias,
        "model_id": model_id,
        "total_samples": total,
        "parsed_success_rate": successes / total if total else 0.0,
        "category_accuracy": category_matches / total if total else 0.0,
        "results": results,
    }


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    token = os.getenv("HF_API_TOKEN")
    if token is None:
        print("⚠️  HF_API_TOKEN not set; requests may be rate-limited", file=sys.stderr)

    records = load_dataset(args.dataset, args.max_samples)
    if not records:
        print("No samples to evaluate.", file=sys.stderr)
        sys.exit(1)

    model_map = {alias: DEFAULT_MODELS.get(alias, alias) for alias in args.models}

    aggregated: Dict[str, Any] = {"dataset_size": len(records), "models": {}}
    for alias, model_id in model_map.items():
        print(f"Evaluating {alias} ({model_id}) on {len(records)} samples...")
        aggregated["models"][alias] = evaluate_model(alias, model_id, records, token)

    args.output.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
