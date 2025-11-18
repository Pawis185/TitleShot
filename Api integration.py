"""
TitleShot - Perception Layer API Integration & Implementation
Author: Shikang Wang
Purpose: Practical implementation with real model APIs
"""

import os
import json
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
from io import BytesIO
import time
from datetime import datetime
import asyncio
import aiohttp
from dataclasses import dataclass, field
import pandas as pd


# ===================== API Configuration =====================
@dataclass
class APIConfig:
    """Configuration for different model APIs"""

    # Qwen2-VL API
    QWEN_API_KEY: str = field(default_factory=lambda: os.getenv("QWEN_API_KEY", ""))
    QWEN_ENDPOINT: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    # InternVL API (through Hugging Face)
    HF_API_KEY: str = field(default_factory=lambda: os.getenv("HF_API_KEY", ""))
    INTERNVL_ENDPOINT: str = "https://api-inference.huggingface.co/models/OpenGVLab/InternVL2-8B"

    # Florence API (through Azure)
    AZURE_API_KEY: str = field(default_factory=lambda: os.getenv("AZURE_API_KEY", ""))
    FLORENCE_ENDPOINT: str = "https://your-resource.cognitiveservices.azure.com/vision/v3.2/analyze"

    # MiniCPM API
    MINICPM_API_KEY: str = field(default_factory=lambda: os.getenv("MINICPM_API_KEY", ""))
    MINICPM_ENDPOINT: str = "https://api.modelbest.cn/v1/chat/completions"

    # BLIP-2 API (through Replicate)
    REPLICATE_API_KEY: str = field(default_factory=lambda: os.getenv("REPLICATE_API_KEY", ""))
    BLIP2_ENDPOINT: str = "https://api.replicate.com/v1/predictions"


# ===================== Model Wrappers =====================
class QwenVLWrapper:
    """Wrapper for Qwen2-VL-7B API"""

    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {self.config.QWEN_API_KEY}",
            "Content-Type": "application/json"
        }

    async def extract_attributes(self, image_url: str, prompt: str) -> Dict:
        """Extract attributes using Qwen2-VL"""

        payload = {
            "model": "qwen-vl-plus",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_url},
                            {"text": prompt}
                        ]
                    }
                ]
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.config.QWEN_ENDPOINT,
                        headers=self.headers,
                        json=payload
                ) as response:
                    result = await response.json()

                    # Parse response
                    if result.get("output", {}).get("choices"):
                        content = result["output"]["choices"][0]["message"]["content"]
                        return self._parse_response(content)
                    else:
                        return {"error": "No response from model"}

        except Exception as e:
            return {"error": str(e)}

    def _parse_response(self, response: str) -> Dict:
        """Parse model response to structured attributes"""
        try:
            # Try to parse as JSON first
            return json.loads(response)
        except:
            # Fallback to text parsing
            attributes = {}
            lines = response.split('\n')

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    attributes[key.strip().lower()] = value.strip()

            return attributes


class InternVLWrapper:
    """Wrapper for InternVL2-8B API"""

    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {self.config.HF_API_KEY}"
        }

    async def extract_attributes(self, image_url: str, prompt: str) -> Dict:
        """Extract attributes using InternVL2"""

        # Download image and convert to base64
        image_data = await self._download_image(image_url)

        payload = {
            "inputs": {
                "image": image_data,
                "text": prompt
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.config.INTERNVL_ENDPOINT,
                        headers=self.headers,
                        json=payload
                ) as response:
                    result = await response.json()
                    return self._parse_response(result)

        except Exception as e:
            return {"error": str(e)}

    async def _download_image(self, url: str) -> str:
        """Download image and convert to base64"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                image_bytes = await response.read()
                return base64.b64encode(image_bytes).decode('utf-8')

    def _parse_response(self, response: Any) -> Dict:
        """Parse model response"""
        if isinstance(response, list) and len(response) > 0:
            return response[0]
        return response


class FlorenceWrapper:
    """Wrapper for Florence-2-Large API"""

    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.config.AZURE_API_KEY,
            "Content-Type": "application/json"
        }

    async def extract_attributes(self, image_url: str, prompt: str) -> Dict:
        """Extract attributes using Florence-2"""

        # Florence uses specific visual features
        params = {
            "visualFeatures": "Categories,Tags,Description,Objects,Brands",
            "language": "en"
        }

        payload = {
            "url": image_url
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.config.FLORENCE_ENDPOINT,
                        headers=self.headers,
                        params=params,
                        json=payload
                ) as response:
                    result = await response.json()
                    return self._parse_response(result, prompt)

        except Exception as e:
            return {"error": str(e)}

    def _parse_response(self, response: Dict, prompt: str) -> Dict:
        """Parse Florence response to our format"""
        attributes = {}

        # Extract categories
        if "categories" in response:
            attributes["category"] = response["categories"][0]["name"] if response["categories"] else "Unknown"

        # Extract tags
        if "tags" in response:
            attributes["tags"] = [tag["name"] for tag in response["tags"][:5]]

        # Extract description
        if "description" in response:
            attributes["description"] = response["description"].get("captions", [{}])[0].get("text", "")

        # Extract objects
        if "objects" in response:
            attributes["objects"] = [obj["object"] for obj in response["objects"]]

        # Extract brands
        if "brands" in response:
            attributes["brand"] = response["brands"][0]["name"] if response["brands"] else None

        return attributes


class MiniCPMWrapper:
    """Wrapper for MiniCPM-V-2.5 API"""

    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {self.config.MINICPM_API_KEY}",
            "Content-Type": "application/json"
        }

    async def extract_attributes(self, image_url: str, prompt: str) -> Dict:
        """Extract attributes using MiniCPM-V"""

        payload = {
            "model": "MiniCPM-V-2_6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.config.MINICPM_ENDPOINT,
                        headers=self.headers,
                        json=payload
                ) as response:
                    result = await response.json()

                    if "choices" in result:
                        content = result["choices"][0]["message"]["content"]
                        return self._parse_response(content)
                    return {"error": "No response from model"}

        except Exception as e:
            return {"error": str(e)}

    def _parse_response(self, response: str) -> Dict:
        """Parse response to structured format"""
        try:
            return json.loads(response)
        except:
            # Basic text parsing
            attributes = {}

            # Common patterns
            patterns = {
                'category': r'Category:\s*([^\n]+)',
                'color': r'Color:\s*([^\n]+)',
                'material': r'Material:\s*([^\n]+)',
                'brand': r'Brand:\s*([^\n]+)',
            }

            import re
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    attributes[key] = match.group(1).strip()

            # Get full description if no structured data
            if not attributes:
                attributes['description'] = response

            return attributes


class BLIP2Wrapper:
    """Wrapper for BLIP-2 API"""

    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.headers = {
            "Authorization": f"Token {self.config.REPLICATE_API_KEY}",
            "Content-Type": "application/json"
        }

    async def extract_attributes(self, image_url: str, prompt: str) -> Dict:
        """Extract attributes using BLIP-2"""

        payload = {
            "version": "version_id_here",  # Replace with actual version
            "input": {
                "image": image_url,
                "prompt": prompt,
                "max_length": 256,
                "temperature": 0.1
            }
        }

        try:
            # Start prediction
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.config.BLIP2_ENDPOINT,
                        headers=self.headers,
                        json=payload
                ) as response:
                    prediction = await response.json()

                    # Poll for results
                    result = await self._poll_for_result(session, prediction["id"])
                    return self._parse_response(result)

        except Exception as e:
            return {"error": str(e)}

    async def _poll_for_result(self, session: aiohttp.ClientSession, prediction_id: str) -> Dict:
        """Poll Replicate API for results"""
        url = f"{self.config.BLIP2_ENDPOINT}/{prediction_id}"

        for _ in range(30):  # Poll for up to 30 seconds
            async with session.get(url, headers=self.headers) as response:
                result = await response.json()

                if result["status"] == "succeeded":
                    return result["output"]
                elif result["status"] == "failed":
                    return {"error": "Prediction failed"}

                await asyncio.sleep(1)

        return {"error": "Timeout waiting for results"}

    def _parse_response(self, response: Any) -> Dict:
        """Parse BLIP-2 response"""
        if isinstance(response, str):
            return {"description": response}
        return response


# ===================== Unified API Interface =====================
class UnifiedPerceptionAPI:
    """Unified interface for all perception models"""

    def __init__(self):
        self.config = APIConfig()

        # Initialize all model wrappers
        self.models = {
            "Qwen2-VL-7B": QwenVLWrapper(self.config),
            "InternVL2-8B": InternVLWrapper(self.config),
            "Florence-2-Large": FlorenceWrapper(self.config),
            "MiniCPM-V-2.5": MiniCPMWrapper(self.config),
            "BLIP-2-OPT-2.7B": BLIP2Wrapper(self.config)
        }

        self.results = []

    async def evaluate_all_models(self,
                                  image_url: str,
                                  prompt: str,
                                  category: str = None) -> pd.DataFrame:
        """Evaluate all models on a single image"""

        print(f"\n{'=' * 60}")
        print(f"Evaluating all models on image")
        print(f"{'=' * 60}")

        tasks = []
        for model_name, model_wrapper in self.models.items():
            print(f"Queuing {model_name}...")
            task = self._evaluate_single_model(
                model_name,
                model_wrapper,
                image_url,
                prompt,
                category
            )
            tasks.append(task)

        # Run all evaluations in parallel
        results = await asyncio.gather(*tasks)

        # Create comparison dataframe
        df = pd.DataFrame(results)
        df = df.sort_values('overall_score', ascending=False)

        return df

    async def _evaluate_single_model(self,
                                     model_name: str,
                                     model_wrapper: Any,
                                     image_url: str,
                                     prompt: str,
                                     category: str) -> Dict:
        """Evaluate a single model"""

        print(f"\nEvaluating {model_name}...")
        start_time = time.time()

        # Get model output
        attributes = await model_wrapper.extract_attributes(image_url, prompt)

        response_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(attributes, category, response_time)

        return {
            'model': model_name,
            'attributes': attributes,
            'response_time': f"{response_time:.2f}s",
            'coverage_score': metrics['coverage'],
            'quality_score': metrics['quality'],
            'consistency_score': metrics['consistency'],
            'overall_score': metrics['overall']
        }

    def _calculate_metrics(self,
                           attributes: Dict,
                           category: str,
                           response_time: float) -> Dict:
        """Calculate evaluation metrics"""

        metrics = {}

        # Coverage: How many expected attributes were extracted
        expected_attrs = ['category', 'color', 'material', 'brand', 'description']
        extracted = sum(1 for attr in expected_attrs if attr in attributes and attributes[attr])
        metrics['coverage'] = extracted / len(expected_attrs)

        # Quality: Based on description length and detail
        description = attributes.get('description', '')
        if description:
            word_count = len(description.split())
            if 50 <= word_count <= 150:
                metrics['quality'] = 1.0
            elif 30 <= word_count < 50:
                metrics['quality'] = 0.7
            else:
                metrics['quality'] = 0.5
        else:
            metrics['quality'] = 0.0

        # Consistency: Check if category matches
        extracted_category = attributes.get('category', '').lower()
        if category and extracted_category:
            metrics['consistency'] = 1.0 if category.lower() in extracted_category else 0.5
        else:
            metrics['consistency'] = 0.5

        # Response time score
        if response_time < 1.0:
            time_score = 1.0
        elif response_time < 3.0:
            time_score = 0.7
        else:
            time_score = 0.4

        # Overall score (weighted average)
        metrics['overall'] = (
                metrics['coverage'] * 0.3 +
                metrics['quality'] * 0.25 +
                metrics['consistency'] * 0.25 +
                time_score * 0.2
        )

        return metrics

    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save evaluation results"""
        # Save as CSV
        df.to_csv(f"{output_path}.csv", index=False)

        # Save detailed JSON
        detailed_results = df.to_dict('records')
        with open(f"{output_path}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\nResults saved to {output_path}.csv and {output_path}.json")


# ===================== Testing Pipeline =====================
class PerceptionLayerTester:
    """Complete testing pipeline for perception layer"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.api = UnifiedPerceptionAPI()
        self.prompts = {
            'structured': """
            Extract the following product attributes in JSON format:
            - Category
            - Primary and secondary colors
            - Material/fabric
            - Brand (if visible)
            - Key features (list 3-5)
            - Target use case
            """,
            'marketing': """
            As an e-commerce expert, analyze this product and provide:
            - Product category and type
            - Visual selling points
            - Quality indicators
            - Target customer profile
            - 3 key marketing messages
            Format as structured data.
            """,
            'detailed': """
            Provide comprehensive product analysis:
            1. Physical attributes (color, size impression, materials)
            2. Functional features
            3. Quality assessment
            4. Suggested usage scenarios
            5. Unique selling propositions
            Be specific and avoid generic descriptions.
            """
        }

    async def test_on_samples(self, num_samples: int = 5):
        """Test models on sample images"""

        # Load dataset
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        all_results = []

        # Test on samples from each category
        for category, items in data.items():
            print(f"\n{'=' * 60}")
            print(f"Testing Category: {category}")
            print(f"{'=' * 60}")

            for i, item in enumerate(items[:num_samples]):
                print(f"\nSample {i + 1}/{min(num_samples, len(items))}")

                # Test with different prompts
                for prompt_name, prompt_text in self.prompts.items():
                    print(f"Using {prompt_name} prompt...")

                    df = await self.api.evaluate_all_models(
                        item['image_url'],
                        prompt_text,
                        category
                    )

                    # Add metadata
                    df['category'] = category
                    df['prompt_type'] = prompt_name
                    df['sample_id'] = i

                    all_results.append(df)

        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)

        # Generate summary statistics
        summary = self._generate_summary(final_df)

        return final_df, summary

    def _generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""

        summary = {
            'model_rankings': {},
            'best_prompt_per_model': {},
            'category_performance': {}
        }

        # Overall model rankings
        model_scores = df.groupby('model')['overall_score'].mean().sort_values(ascending=False)
        summary['model_rankings'] = model_scores.to_dict()

        # Best prompt for each model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_prompt = model_df.groupby('prompt_type')['overall_score'].mean().idxmax()
            summary['best_prompt_per_model'][model] = best_prompt

        # Performance by category
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            best_model = cat_df.groupby('model')['overall_score'].mean().idxmax()
            summary['category_performance'][category] = best_model

        return summary

    def print_summary(self, summary: Dict):
        """Print formatted summary"""

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print("\n📊 Overall Model Rankings:")
        for i, (model, score) in enumerate(summary['model_rankings'].items(), 1):
            print(f"  {i}. {model}: {score:.3f}")

        print("\n🎯 Best Prompt Template per Model:")
        for model, prompt in summary['best_prompt_per_model'].items():
            print(f"  {model}: {prompt}")

        print("\n📦 Best Model per Category:")
        for category, model in summary['category_performance'].items():
            print(f"  {category}: {model}")


# ===================== Main Execution =====================
async def main():
    """Main execution function"""

    print("=" * 80)
    print("TITLESHOT - PERCEPTION LAYER API TESTING")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")

    # Initialize tester
    tester = PerceptionLayerTester("dataset.json")

    # Run tests
    print("\n🚀 Starting model evaluation...")
    results_df, summary = await tester.test_on_samples(num_samples=2)

    # Save results
    output_path = "/home/claude/api_test_results"
    tester.api.save_results(results_df, output_path)

    # Print summary
    tester.print_summary(summary)

    # Identify best model
    best_model = max(summary['model_rankings'].items(), key=lambda x: x[1])

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"✨ Best Overall Model: {best_model[0]}")
    print(f"   Score: {best_model[1]:.3f}")

    print("\n📝 Implementation Steps:")
    print("1. Set up API credentials for", best_model[0])
    print("2. Use the", summary['best_prompt_per_model'][best_model[0]], "prompt template")
    print("3. Deploy lightweight enhancement model for weak attributes")
    print("4. Implement caching for common product types")
    print("5. Set up monitoring and continuous evaluation")

    return results_df, summary


# ===================== Script Execution =====================
if __name__ == "__main__":
    # Note: This is for demonstration. In practice, you'd run:
    # asyncio.run(main())

    print("\n" + "=" * 80)
    print("API INTEGRATION READY")
    print("=" * 80)
    print("\n⚡ To run the evaluation:")
    print("   import asyncio")
    print("   asyncio.run(main())")

    print("\n🔧 Environment variables needed:")
    print("   - QWEN_API_KEY")
    print("   - HF_API_KEY (for Hugging Face)")
    print("   - AZURE_API_KEY (for Florence)")
    print("   - MINICPM_API_KEY")
    print("   - REPLICATE_API_KEY (for BLIP-2)")

    print("\n📚 API Documentation:")
    print("   - Qwen: https://help.aliyun.com/document_detail/2400395.html")
    print("   - Hugging Face: https://huggingface.co/docs/api-inference")
    print("   - Azure Vision: https://docs.microsoft.com/en-us/azure/cognitive-services/")
    print("   - MiniCPM: https://github.com/OpenBMB/MiniCPM-V")
    print("   - Replicate: https://replicate.com/docs")