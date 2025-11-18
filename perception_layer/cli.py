"""Command line interface for the TitleShot perception pipeline.

This module provides convenient CLI tools for:
- Single image extraction
- Batch processing
- Model comparison
- Benchmarking
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import requests

from .model import (
    ModelType,
    PerceptionPipeline,
    quick_extract,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TitleShot Perception Layer - E-commerce Image Attribute Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract attributes from a single image
  python -m perception_layer.cli image.jpg
  
  # Use Qwen2-VL model
  python -m perception_layer.cli image.jpg --model qwen2-vl-7b
  
  # Add text hint and save output
  python -m perception_layer.cli image.jpg --text "Cotton T-shirt" --output result.json
  
  # Compare multiple models
  python -m perception_layer.cli image.jpg --compare --models qwen2-vl-7b florence2-large
  
  # Batch process directory
  python -m perception_layer.cli --batch ./images/ --output-dir ./results/
  
  # Benchmark models
  python -m perception_layer.cli --benchmark ./test_images/ --models qwen2-vl-7b blip2-flan-t5-xl
  
  # Show model recommendations
  python -m perception_layer.cli --recommend
        """,
    )

    # Main input
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        help="Path to the product image (or directory for --batch mode)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="blip2-flan-t5-xl",
        choices=[m.value for m in ModelType],
        help="Vision-language model to use (default: blip2-flan-t5-xl)",
    )

    # Text input
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Optional textual hint (title, bullet point, etc.)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save the JSON output (default: print to stdout)",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "pretty", "generation"],
        default="pretty",
        help="Output format (json=compact, pretty=formatted, generation=for LLM input)",
    )

    # Batch processing
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all images in a directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save batch results (creates one JSON per image)",
    )

    # Model comparison
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare multiple models on the same image",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m.value for m in ModelType],
        default=None,
        help="List of models to compare (requires --compare)",
    )

    # Benchmarking
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on a directory of images",
    )

    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Path to dataset.json containing image URLs for benchmarking",
    )

    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=3,
        help="Number of images to download per category when using --dataset-json",
    )

    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(".cache/dataset_images"),
        help="Directory to cache downloaded dataset images",
    )

    # Utilities
    parser.add_argument(
        "--recommend",
        "-r",
        action="store_true",
        help="Show model recommendations and exit",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run models on (default: auto-detect)",
    )

    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Use multilingual text encoder for Chinese support",
    )

    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Disable image quality checks",
    )

    return parser.parse_args()


def format_output(result_dict: dict, format_type: str) -> str:
    """Format output based on format type.

    Args:
        result_dict: Result dictionary from perception pipeline
        format_type: One of 'json', 'pretty', 'generation'

    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(result_dict, ensure_ascii=False)

    elif format_type == "generation":
        # Format for generation layer input
        attrs = result_dict.get("image_attributes", {})
        output = "Product Attributes:\n"
        output += f"- Category: {attrs.get('category', 'N/A')}\n"
        output += f"- Color: {attrs.get('color', 'N/A')}\n"
        output += f"- Material: {attrs.get('material', 'N/A')}\n"
        output += f"- Style: {attrs.get('style', 'N/A')}\n"

        if "selling_points" in attrs:
            points = attrs.get("selling_points", [])
            output += f"- Selling Points: {', '.join(points)}\n"

        if result_dict.get("keywords"):
            output += f"- Keywords: {', '.join(result_dict['keywords'])}\n"

        return output

    else:  # pretty
        return json.dumps(result_dict, indent=2, ensure_ascii=False)


def download_images_from_dataset(
    dataset_path: Path,
    download_dir: Path,
    limit_per_category: int = 3,
) -> List[Path]:
    """Download images listed in a dataset JSON file.

    Args:
        dataset_path: Path to dataset.json
        download_dir: Directory where downloaded images will be cached
        limit_per_category: How many images to pull per category

    Returns:
        List of local image paths that were downloaded or already cached
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    download_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []

    for category, items in data.items():
        safe_category = re.sub(r"[^a-zA-Z0-9_-]+", "_", category)
        for idx, item in enumerate(items[:limit_per_category], start=1):
            url = item.get("image_url")
            if not url:
                continue

            ext = Path(url).suffix or ".jpg"
            filename = f"{safe_category}_{idx}{ext}"
            target_path = download_dir / filename

            if not target_path.exists():
                try:
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    target_path.write_bytes(response.content)
                    print(f"⬇️  Downloaded {url} → {target_path}")
                except Exception as exc:  # noqa: BLE001
                    print(f"⚠️  Skipping {url}: {exc}")
                    continue

            image_paths.append(target_path)

    return image_paths


def handle_single_extraction(args: argparse.Namespace) -> None:
    """Handle single image extraction."""
    if not args.image:
        print("❌ Error: Image path required", file=sys.stderr)
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Extracting attributes from: {image_path.name}")

    try:
        result_dict = quick_extract(
            image_path,
            model_type=ModelType(args.model),
            text_hint=args.text,
            verbose=False,
        )

        output = format_output(result_dict, args.format)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
            print(f"💾 Results saved to: {args.output}")
        else:
            print("\n" + "=" * 70)
            print(output)
            print("=" * 70)

        # Show summary
        attrs = result_dict.get("image_attributes", {})
        print(f"\n✅ Extraction complete in {result_dict.get('extraction_time', 0):.2f}s")
        print(f"📊 Attributes extracted: {len(attrs)}")

    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def handle_batch_processing(args: argparse.Namespace) -> None:
    """Handle batch directory processing."""
    if not args.image:
        print("❌ Error: Directory path required for batch mode", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.image)
    if not input_dir.is_dir():
        print(f"❌ Error: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print(f"❌ Error: No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"📦 Processing {len(image_paths)} images from {input_dir}")

    # Create pipeline
    pipeline = PerceptionPipeline(
        model_type=ModelType(args.model),
        device=args.device,
        use_multilingual=args.multilingual,
    )

    # Process batch
    results = pipeline.batch_run(image_paths, show_progress=True)

    # Save results
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        for image_path, result in zip(image_paths, results):
            output_path = args.output_dir / f"{image_path.stem}_result.json"
            output_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print(f"\n💾 All results saved to: {args.output_dir}")

    # Show summary
    successful = sum(1 for r in results if r.image_attributes.attributes)
    print(f"\n✅ Batch processing complete:")
    print(f"   - Total images: {len(image_paths)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {len(image_paths) - successful}")


def handle_model_comparison(args: argparse.Namespace) -> None:
    """Handle model comparison."""
    if not args.image:
        print("❌ Error: Image path required for comparison", file=sys.stderr)
        sys.exit(1)

    if not args.models:
        print("❌ Error: --models required for comparison", file=sys.stderr)
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    model_types = [ModelType(m) for m in args.models]

    pipeline = PerceptionPipeline(device=args.device)
    results = pipeline.compare_models(image_path, model_types, text_hint=args.text)

    # Show comparison table
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 70)

    for model_name, result in results.items():
        if result is None:
            print(f"\n❌ {model_name}: FAILED")
            continue

        print(f"\n✅ {model_name}:")
        print(f"   Time: {result.image_attributes.extraction_time:.2f}s")
        print(f"   Attributes: {result.image_attributes.attributes}")

    print("=" * 70)


def handle_benchmark(args: argparse.Namespace) -> None:
    """Handle benchmarking."""
    if not args.models:
        print("❌ Error: --models required for benchmark", file=sys.stderr)
        sys.exit(1)

    image_paths: List[Path] = []

    if args.dataset_json:
        image_paths = download_images_from_dataset(
            args.dataset_json,
            args.download_dir,
            limit_per_category=args.limit_per_category,
        )
    elif args.image:
        input_dir = Path(args.image)
        if not input_dir.is_dir():
            print(f"❌ Error: Not a directory: {input_dir}", file=sys.stderr)
            sys.exit(1)

        # Find images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]

    if not image_paths:
        print("❌ Error: No images available for benchmarking", file=sys.stderr)
        sys.exit(1)

    model_types = [ModelType(m) for m in args.models]

    pipeline = PerceptionPipeline(device=args.device)
    benchmark_results = pipeline.benchmark_models(image_paths, model_types)

    # Print results table
    print("\n" + "=" * 70)
    print("🏆 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Avg Time':<12} {'Success':<10} {'Complete':<10}")
    print("-" * 70)

    for model_name, metrics in benchmark_results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['avg_extraction_time']:>6.2f}s     "
            f"{metrics['success_rate']:>5.1f}%     "
            f"{metrics['attribute_completeness']:>5.1f}%"
        )

    print("=" * 70)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(benchmark_results, indent=2),
            encoding="utf-8",
        )
        print(f"\n💾 Benchmark results saved to: {args.output}")


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for CLI."""
    parsed_args = args or parse_args()

    # Handle special commands
    if parsed_args.recommend:
        PerceptionPipeline.print_model_recommendations()
        return

    # Route to appropriate handler
    try:
        if parsed_args.benchmark:
            handle_benchmark(parsed_args)
        elif parsed_args.compare:
            handle_model_comparison(parsed_args)
        elif parsed_args.batch:
            handle_batch_processing(parsed_args)
        else:
            handle_single_extraction(parsed_args)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()