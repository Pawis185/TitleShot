"""Test script for the perception layer.

This script demonstrates how to use the perception layer and validates
the integration with the generation layer (Yiyang Wang's module).
"""
import json
from pathlib import Path

from perception_layer.model import (
    ModelType,
    PerceptionPipeline,
    quick_extract,
)


def test_single_extraction():
    """Test 1: Single image extraction with different models."""
    print("\n" + "=" * 70)
    print("TEST 1: Single Image Extraction")
    print("=" * 70)

    # You'll need to replace this with an actual image path
    test_image = "sample_product.jpg"

    if not Path(test_image).exists():
        print(f"⚠️  Skipping: {test_image} not found")
        print("   Please provide a test image to run this test")
        return

    # Test with BLIP-2 (baseline)
    print("\n📸 Testing with BLIP-2...")
    result = quick_extract(
        test_image,
        model_type=ModelType.BLIP2_FLAN_T5_XL,
        text_hint="Premium leather wallet",
        verbose=True,
    )

    print(f"✅ Extraction completed successfully")
    print(f"   Extracted {len(result['image_attributes'])} attributes")


def test_model_comparison():
    """Test 2: Compare multiple models on the same image."""
    print("\n" + "=" * 70)
    print("TEST 2: Model Comparison")
    print("=" * 70)

    test_image = "sample_product.jpg"

    if not Path(test_image).exists():
        print(f"⚠️  Skipping: {test_image} not found")
        return

    # Compare lightweight models
    models_to_compare = [
        ModelType.BLIP2_OPT_2_7B,  # Baseline
        ModelType.FLORENCE2_LARGE,  # Fast
        # Add more if you have enough VRAM:
        # ModelType.QWEN2_VL_7B,
        # ModelType.MINICPM_V_2_5,
    ]

    pipeline = PerceptionPipeline()
    results = pipeline.compare_models(test_image, models_to_compare)

    print("\n📊 Comparison Summary:")
    for model_name, result in results.items():
        if result:
            print(f"   {model_name}: {result.image_attributes.extraction_time:.2f}s")


def test_batch_processing():
    """Test 3: Batch processing multiple images."""
    print("\n" + "=" * 70)
    print("TEST 3: Batch Processing")
    print("=" * 70)

    # Create a test directory with some images
    test_dir = Path("test_images")

    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"⚠️  Skipping: {test_dir} not found or empty")
        print("   Create a 'test_images' directory with some product images")
        return

    image_paths = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    print(f"📦 Found {len(image_paths)} images")

    pipeline = PerceptionPipeline(model_type=ModelType.BLIP2_OPT_2_7B)
    results = pipeline.batch_run(image_paths[:5], show_progress=True)  # Test first 5

    print(f"\n✅ Processed {len(results)} images")
    successful = sum(1 for r in results if r.image_attributes.attributes)
    print(f"   Success rate: {successful}/{len(results)}")


def test_generation_layer_integration():
    """Test 4: Output format for generation layer integration."""
    print("\n" + "=" * 70)
    print("TEST 4: Generation Layer Integration")
    print("=" * 70)

    test_image = "sample_product.jpg"

    if not Path(test_image).exists():
        print(f"⚠️  Skipping: {test_image} not found")
        return

    # Extract attributes
    pipeline = PerceptionPipeline(model_type=ModelType.BLIP2_OPT_2_7B)
    result = pipeline.run(test_image, text_hint="Comfortable running shoes")

    # Format 1: Full dictionary (for API communication)
    print("\n📤 Format 1: Full Dictionary (for API)")
    print("-" * 70)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    # Format 2: Structured prompt (for LLM input)
    print("\n📤 Format 2: Structured Prompt (for LLM)")
    print("-" * 70)
    print(result.to_generation_input())

    print("\n✅ Both formats ready for Yiyang Wang's generation layer")


def test_quality_checks():
    """Test 5: Image quality validation."""
    print("\n" + "=" * 70)
    print("TEST 5: Image Quality Checks")
    print("=" * 70)

    # Test with a low-quality image
    test_cases = [
        ("sample_product.jpg", "Normal quality"),
        ("low_res_product.jpg", "Low resolution (should warn)"),
        ("dark_product.jpg", "Too dark (should warn)"),
    ]

    for image_path, description in test_cases:
        if not Path(image_path).exists():
            continue

        print(f"\n🔍 Testing: {description}")
        try:
            pipeline = PerceptionPipeline(
                model_type=ModelType.BLIP2_OPT_2_7B,
            )
            result = pipeline.run(image_path)

            if result.image_attributes.image_quality:
                quality = result.image_attributes.image_quality
                print(f"   Size: {quality.width}x{quality.height}")
                print(f"   Brightness: {quality.mean_brightness:.1f}")
                print(f"   Acceptable: {quality.is_acceptable}")

        except Exception as e:
            print(f"   ❌ {str(e)}")


def test_benchmark():
    """Test 6: Quick benchmark on a small dataset."""
    print("\n" + "=" * 70)
    print("TEST 6: Model Benchmarking")
    print("=" * 70)

    test_dir = Path("test_images")

    if not test_dir.exists():
        print(f"⚠️  Skipping: {test_dir} not found")
        return

    image_paths = list(test_dir.glob("*.jpg"))[:10]  # Test on 10 images

    if len(image_paths) < 3:
        print("⚠️  Need at least 3 images for meaningful benchmark")
        return

    # Benchmark fast models only
    models_to_test = [
        ModelType.BLIP2_OPT_2_7B,
        ModelType.FLORENCE2_LARGE,
    ]

    pipeline = PerceptionPipeline()
    results = pipeline.benchmark_models(image_paths, models_to_test)

    print("\n📊 Benchmark completed - see results above")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 TITLESHOT PERCEPTION LAYER TEST SUITE")
    print("=" * 70)

    tests = [
        ("Single Extraction", test_single_extraction),
        ("Model Comparison", test_model_comparison),
        ("Batch Processing", test_batch_processing),
        ("Generation Layer Integration", test_generation_layer_integration),
        ("Quality Checks", test_quality_checks),
        ("Benchmarking", test_benchmark),
    ]

    print("\nAvailable tests:")
    for idx, (name, _) in enumerate(tests, 1):
        print(f"  {idx}. {name}")

    print("\n" + "=" * 70)

    # Run all tests
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' failed: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ TEST SUITE COMPLETED")
    print("=" * 70)

    # Print recommendations
    print("\n💡 Next Steps:")
    print("   1. Replace 'sample_product.jpg' with real Amazon product images")
    print("   2. Create a 'test_images' directory with more test cases")
    print("   3. Try different models based on your VRAM availability")
    print("   4. Share the output format with Yiyang Wang for integration")
    print("\n🔧 To see model recommendations, run:")
    print("   python -m perception_layer.cli --recommend")


if __name__ == "__main__":
    main()