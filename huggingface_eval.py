"""Run lightweight evaluations against Hugging Face Inference API.

Improvements:
1. Added retry logic for rate limiting
2. Better error handling
3. Progress bar support
4. Support for alternative API endpoints

Usage:
    # 设置 token (推荐)
    export HF_API_TOKEN=hf_xxxxx

    # 或者直接在命令行传入
    python improved_huggingface_eval.py \
        --dataset dataset.json \
        --models qwen2-vl-7b internvl2-8b llava-1.5-7b \
        --output hf_results.json \
        --token hf_xxxxx \
        --max-samples 20
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

try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed. Install it with:")
    print("  pip install huggingface_hub")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


DEFAULT_MODELS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",
    "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b",
    "minicpm-v-2.5": "openbmb/MiniCPM-Llama3-V-2_5",
    "florence2-large": "microsoft/Florence-2-large",
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
        default=["qwen2-vl-7b", "internvl2-8b", "llava-1.5-7b"],
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
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_API_TOKEN env var)",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts for failed requests",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay between retries in seconds",
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
    """Parse JSON block from model output."""
    # Try direct JSON parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block with ```json markers
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.S)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    json_match = re.search(r"\{.*?\}", content, flags=re.S)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

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
    """Build extraction prompt."""
    return (
        "You are an expert e-commerce analyst. Identify the product attributes "
        "from the image and respond ONLY with JSON using the keys: category, "
        "color, material, style, selling_points.\n\n"
        "Category should be one of: Electronics, Home_and_Kitchen, Books, "
        "Clothing_Shoes_and_Jewelry, Sports_and_Outdoors, Movies_and_TV, "
        "Automotive, Tools_and_Home_Improvement, Pet_Supplies, "
        "Health_and_Personal_Care, Toys_and_Games, Office_Products.\n\n"
        "Selling_points should be a list of 2 key product features.\n\n"
        "Respond with valid JSON only, no additional text."
    )


def evaluate_model(
    model_alias: str,
    model_id: str,
    records: Iterable[Dict[str, str]],
    token: Optional[str] = None,
    retry_attempts: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate a single model and return aggregate metrics and raw outputs."""
    client = InferenceClient(token=token)

    results: List[Dict[str, Any]] = []
    successes = 0
    category_matches = 0

    records_list = list(records)
    iterator = tqdm(records_list, desc=f"Evaluating {model_alias}") if HAS_TQDM else records_list

    for record in iterator:
        prompt = _build_prompt(record["category"])

        # Retry logic
        for attempt in range(retry_attempts):
            try:
                start = time.perf_counter()

                # Make API call
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

                # Extract response
                content = _coerce_content_to_text(completion.choices[0].message["content"])
                parsed = _extract_json_from_response(content)

                is_success = parsed is not None
                predicted_category = (parsed or {}).get("category", "")

                # Normalize category for comparison
                expected_normalized = record["category"].lower().replace("_", " ")
                predicted_normalized = predicted_category.lower().replace("_", " ")
                matches_category = bool(
                    predicted_category
                    and (expected_normalized in predicted_normalized or predicted_normalized in expected_normalized)
                )

                successes += int(is_success)
                category_matches += int(matches_category)

                results.append(
                    {
                        "image_url": record["image_url"],
                        "expected_category": record["category"],
                        "predicted_category": predicted_category,
                        "raw_response": content,
                        "parsed": parsed,
                        "latency_sec": latency,
                        "parsed_ok": is_success,
                        "category_match": matches_category,
                        "attempt": attempt + 1,
                    }
                )

                # Success, break retry loop
                break

            except HfHubHTTPError as err:
                if attempt < retry_attempts - 1:
                    if HAS_TQDM:
                        tqdm.write(f"⚠️  Rate limited, retrying in {retry_delay}s... (attempt {attempt + 1}/{retry_attempts})")
                    time.sleep(retry_delay)
                    continue
                else:
                    results.append(
                        {
                            "image_url": record["image_url"],
                            "expected_category": record["category"],
                            "error": str(err),
                            "error_type": "rate_limit",
                            "latency_sec": None,
                            "parsed_ok": False,
                            "category_match": False,
                        }
                    )

            except Exception as exc:
                if attempt < retry_attempts - 1:
                    if HAS_TQDM:
                        tqdm.write(f"⚠️  Error: {str(exc)}, retrying... (attempt {attempt + 1}/{retry_attempts})")
                    time.sleep(retry_delay)
                    continue
                else:
                    results.append(
                        {
                            "image_url": record["image_url"],
                            "expected_category": record["category"],
                            "error": str(exc),
                            "error_type": "unknown",
                            "latency_sec": None,
                            "parsed_ok": False,
                            "category_match": False,
                        }
                    )

    total = len(results)
    avg_latency = sum(r.get("latency_sec", 0) or 0 for r in results) / successes if successes > 0 else 0

    return {
        "model_alias": model_alias,
        "model_id": model_id,
        "total_samples": total,
        "successful_requests": successes,
        "parsed_success_rate": successes / total if total else 0.0,
        "category_accuracy": category_matches / total if total else 0.0,
        "avg_latency_sec": avg_latency,
        "results": results,
    }


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Get token from args or environment
    token = args.token or os.getenv("HF_API_TOKEN")
    if token is None:
        print("⚠️  No HuggingFace token provided!", file=sys.stderr)
        print("   Set HF_API_TOKEN environment variable or use --token argument", file=sys.stderr)
        print("   Requests may be rate-limited without authentication", file=sys.stderr)
        print()

    # Load dataset
    records = load_dataset(args.dataset, args.max_samples)
    if not records:
        print("❌ No samples to evaluate.", file=sys.stderr)
        sys.exit(1)

    print(f"📊 Loaded {len(records)} samples from {args.dataset}")
    print(f"🤖 Evaluating {len(args.models)} models")
    print()

    # Map model aliases to full IDs
    model_map = {}
    for alias in args.models:
        if alias in DEFAULT_MODELS:
            model_map[alias] = DEFAULT_MODELS[alias]
        else:
            # Assume it's a full model ID
            model_map[alias] = alias

    # Run evaluations
    aggregated: Dict[str, Any] = {
        "dataset_size": len(records),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }

    for alias, model_id in model_map.items():
        print(f"\n{'='*70}")
        print(f"🔍 Evaluating: {alias}")
        print(f"   Model ID: {model_id}")
        print(f"{'='*70}")

        result = evaluate_model(
            alias,
            model_id,
            records,
            token,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay,
        )

        aggregated["models"][alias] = result

        # Print summary
        print(f"\n📈 Results for {alias}:")
        print(f"   ✅ Success Rate: {result['parsed_success_rate']*100:.1f}%")
        print(f"   🎯 Category Accuracy: {result['category_accuracy']*100:.1f}%")
        print(f"   ⏱️  Avg Latency: {result['avg_latency_sec']:.2f}s")
        print(f"   📊 Successful: {result['successful_requests']}/{result['total_samples']}")

    # Save results
    args.output.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"💾 Results saved to: {args.output.resolve()}")
    print(f"{'='*70}")

    # Print comparison table
    print(f"\n📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Success Rate':<15} {'Category Acc':<15} {'Avg Latency':<12}")
    print(f"{'-'*70}")

    for alias, result in aggregated["models"].items():
        print(
            f"{alias:<20} "
            f"{result['parsed_success_rate']*100:>6.1f}%        "
            f"{result['category_accuracy']*100:>6.1f}%        "
            f"{result['avg_latency_sec']:>6.2f}s"
        )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()