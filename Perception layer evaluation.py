"""
TitleShot - Perception Layer Evaluation Framework
Author: Shikang Wang
Purpose: Evaluate and compare vision-language models for e-commerce product attribute extraction
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import os
from dataclasses import dataclass, asdict
from collections import Counter
import re

# ===================== Configuration =====================
@dataclass
class ModelConfig:
    """Configuration for each model to be tested"""
    name: str
    api_endpoint: str = None
    local_path: str = None
    model_type: str = "api"  # "api" or "local"
    max_tokens: int = 256
    temperature: float = 0.1

# Models to evaluate
MODELS_TO_EVALUATE = [
    ModelConfig(name="Qwen2-VL-7B", model_type="local", local_path="Qwen/Qwen2-VL-7B"),
    ModelConfig(name="InternVL2-8B", model_type="local", local_path="OpenGVLab/InternVL2-8B"),
    ModelConfig(name="Florence-2-Large", model_type="local", local_path="microsoft/Florence-2-large"),
    ModelConfig(name="MiniCPM-V-2.5", model_type="local", local_path="openbmb/MiniCPM-V-2_6"),
    ModelConfig(name="BLIP-2-OPT-2.7B", model_type="local", local_path="Salesforce/blip2-opt-2.7b"),
]

# ===================== Evaluation Metrics =====================
class EvaluationMetrics:
    """Define and calculate all evaluation metrics for perception layer"""

    @staticmethod
    def attribute_extraction_accuracy(predicted: Dict, ground_truth: Dict) -> float:
        """
        Measure how accurately the model extracts product attributes
        """
        correct = 0
        total = 0

        for key in ground_truth:
            if key in predicted:
                if predicted[key].lower() == ground_truth[key].lower():
                    correct += 1
            total += 1

        return correct / total if total > 0 else 0

    @staticmethod
    def attribute_coverage(predicted: Dict, expected_attributes: List[str]) -> float:
        """
        Measure the percentage of expected attributes that were extracted
        """
        extracted = sum(1 for attr in expected_attributes if attr in predicted)
        return extracted / len(expected_attributes) if expected_attributes else 0

    @staticmethod
    def description_quality_score(description: str) -> Dict[str, float]:
        """
        Evaluate the quality of generated descriptions
        """
        scores = {}

        # Length appropriateness (optimal: 50-150 words)
        word_count = len(description.split())
        if 50 <= word_count <= 150:
            scores['length_score'] = 1.0
        elif 30 <= word_count < 50 or 150 < word_count <= 200:
            scores['length_score'] = 0.7
        else:
            scores['length_score'] = 0.3

        # Marketing keywords presence
        marketing_keywords = ['premium', 'quality', 'perfect', 'ideal', 'durable',
                             'comfortable', 'stylish', 'innovative', 'efficient', 'versatile']
        keyword_count = sum(1 for kw in marketing_keywords if kw.lower() in description.lower())
        scores['marketing_appeal'] = min(keyword_count / 3, 1.0)  # Normalize to 0-1

        # Specificity (presence of specific details vs generic terms)
        specific_patterns = [r'\d+', r'[A-Z]{2,}', r'\b(model|version|series)\b']
        specificity_count = sum(1 for pattern in specific_patterns if re.search(pattern, description))
        scores['specificity'] = min(specificity_count / 2, 1.0)

        return scores

    @staticmethod
    def semantic_consistency(attributes: Dict, description: str) -> float:
        """
        Check if the description is consistent with extracted attributes
        """
        consistency_score = 0
        total_attributes = 0

        for attr_key, attr_value in attributes.items():
            if isinstance(attr_value, str) and attr_value.lower() in description.lower():
                consistency_score += 1
            total_attributes += 1

        return consistency_score / total_attributes if total_attributes > 0 else 0

    @staticmethod
    def response_time_efficiency(response_time: float) -> float:
        """
        Evaluate response time efficiency (optimal < 1s, acceptable < 3s)
        """
        if response_time < 1.0:
            return 1.0
        elif response_time < 3.0:
            return 0.7
        elif response_time < 5.0:
            return 0.4
        else:
            return 0.1

    @staticmethod
    def hallucination_detection(predicted_attrs: Dict, image_category: str) -> float:
        """
        Detect potential hallucinations based on category consistency
        """
        # Define expected attributes per category
        category_attributes = {
            'Electronics': ['brand', 'model', 'color', 'connectivity', 'power', 'display'],
            'Clothing': ['size', 'color', 'material', 'style', 'pattern', 'season'],
            'Home & Kitchen': ['material', 'color', 'size', 'capacity', 'features', 'brand'],
            'Books': ['title', 'author', 'genre', 'format', 'language', 'pages'],
            'Sports & Outdoors': ['brand', 'size', 'material', 'color', 'sport_type', 'features']
        }

        if image_category not in category_attributes:
            return 0.5  # Neutral score for unknown categories

        expected = category_attributes[image_category]
        relevant_attrs = sum(1 for attr in predicted_attrs.keys()
                           if any(exp in attr.lower() for exp in expected))
        irrelevant_attrs = len(predicted_attrs) - relevant_attrs

        # Penalize irrelevant attributes
        return max(0, 1 - (irrelevant_attrs * 0.2))

# ===================== Model Interface =====================
class VisionLanguageModelInterface:
    """Unified interface for different vision-language models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the model based on configuration"""
        if self.config.model_type == "local":
            print(f"Loading {self.config.name} locally...")
            # Placeholder for actual model loading
            # This would be replaced with actual model loading code
            # Example for BLIP-2:
            # from transformers import Blip2Processor, Blip2ForConditionalGeneration
            # self.processor = Blip2Processor.from_pretrained(self.config.local_path)
            # self.model = Blip2ForConditionalGeneration.from_pretrained(self.config.local_path)
            pass
        else:
            print(f"Configuring API access for {self.config.name}...")
            pass

    def extract_attributes(self, image_url: str, prompt_template: str) -> Tuple[Dict, float]:
        """
        Extract attributes from image using the model
        Returns: (attributes_dict, response_time)
        """
        start_time = time.time()

        # Placeholder for actual model inference
        # This is where you'd implement actual model calls
        attributes = self._mock_extraction(image_url, prompt_template)

        response_time = time.time() - start_time
        return attributes, response_time

    def _mock_extraction(self, image_url: str, prompt_template: str) -> Dict:
        """
        Mock extraction for demonstration
        Replace with actual model inference
        """
        # Simulate different model behaviors
        if "BLIP" in self.config.name:
            return {
                "category": "Electronics",
                "color": "Black",
                "brand": "Generic",
                "material": "Plastic",
                "features": ["wireless", "portable"],
                "description": "A sleek electronic device with modern design and wireless connectivity"
            }
        elif "Qwen" in self.config.name:
            return {
                "category": "Electronics",
                "color": "Black/Silver",
                "brand": "Tech Brand",
                "model": "Model X",
                "connectivity": "Bluetooth 5.0",
                "description": "Premium electronic device featuring advanced connectivity and durable construction"
            }
        else:
            return {
                "category": "Electronics",
                "primary_color": "Black",
                "secondary_color": "Silver",
                "key_features": "Wireless, Compact, Energy-efficient",
                "description": "Innovative electronic product designed for modern lifestyle"
            }

# ===================== Prompt Engineering =====================
class PromptTemplates:
    """Collection of optimized prompts for attribute extraction"""

    STRUCTURED_EXTRACTION = """
    Analyze this product image and extract the following structured information:
    1. Product Category
    2. Primary Color
    3. Material (if visible)
    4. Brand (if visible)
    5. Key Features (list up to 5)
    6. Target Usage Scenario
    7. Unique Selling Points
    
    Provide the output in JSON format.
    """

    MARKETING_FOCUSED = """
    As an e-commerce expert, analyze this product image and provide:
    - Category and subcategory
    - Visual attributes (color, shape, size impression)
    - Quality indicators (materials, craftsmanship)
    - Marketing appeal factors
    - Suggested target audience
    - Key selling points for product listing
    
    Format as structured data suitable for e-commerce listing.
    """

    DETAILED_DESCRIPTION = """
    Generate a comprehensive product analysis including:
    - Physical attributes
    - Functional features
    - Quality assessment
    - Usage scenarios
    - Competitive advantages
    - Marketing highlights
    
    Be specific and avoid generic descriptions.
    """

    @classmethod
    def get_all_templates(cls) -> Dict[str, str]:
        return {
            "structured": cls.STRUCTURED_EXTRACTION,
            "marketing": cls.MARKETING_FOCUSED,
            "detailed": cls.DETAILED_DESCRIPTION
        }

# ===================== Evaluation Pipeline =====================
class PerceptionLayerEvaluator:
    """Main evaluation pipeline for perception layer models"""

    def __init__(self, dataset_path: str, output_dir: str = "./evaluation_results"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.metrics = EvaluationMetrics()
        self.prompts = PromptTemplates()
        self.results = []

    def load_dataset(self, sample_size: int = None) -> List[Dict]:
        """Load and prepare the dataset"""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        # Flatten the dataset
        samples = []
        for category, items in data.items():
            for item in items[:sample_size] if sample_size else items:
                item['ground_truth_category'] = category
                samples.append(item)

        return samples

    def evaluate_model(self, model_config: ModelConfig, samples: List[Dict]) -> Dict:
        """Evaluate a single model across all samples"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_config.name}")
        print(f"{'='*60}")

        model_interface = VisionLanguageModelInterface(model_config)
        model_interface.load_model()

        model_results = {
            'model_name': model_config.name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': {},
            'sample_results': []
        }

        # Aggregate metrics
        all_accuracy_scores = []
        all_coverage_scores = []
        all_quality_scores = []
        all_consistency_scores = []
        all_response_times = []
        all_hallucination_scores = []

        # Test with different prompts
        for prompt_name, prompt_template in self.prompts.get_all_templates().items():
            print(f"\nTesting with {prompt_name} prompt template...")

            for idx, sample in enumerate(samples):
                if idx % 10 == 0:
                    print(f"Processing sample {idx+1}/{len(samples)}...")

                # Extract attributes
                attributes, response_time = model_interface.extract_attributes(
                    sample['image_url'],
                    prompt_template
                )

                # Calculate metrics
                expected_attrs = ['category', 'color', 'material', 'brand', 'features']

                coverage = self.metrics.attribute_coverage(attributes, expected_attrs)
                quality = self.metrics.description_quality_score(
                    attributes.get('description', '')
                )
                consistency = self.metrics.semantic_consistency(
                    attributes,
                    attributes.get('description', '')
                )
                efficiency = self.metrics.response_time_efficiency(response_time)
                hallucination = self.metrics.hallucination_detection(
                    attributes,
                    sample['ground_truth_category']
                )

                # Store results
                sample_result = {
                    'sample_id': idx,
                    'prompt_type': prompt_name,
                    'extracted_attributes': attributes,
                    'metrics': {
                        'coverage': coverage,
                        'quality_scores': quality,
                        'consistency': consistency,
                        'response_time': response_time,
                        'efficiency': efficiency,
                        'hallucination_score': hallucination
                    }
                }

                model_results['sample_results'].append(sample_result)

                # Aggregate
                all_coverage_scores.append(coverage)
                all_quality_scores.append(quality)
                all_consistency_scores.append(consistency)
                all_response_times.append(response_time)
                all_hallucination_scores.append(hallucination)

        # Calculate aggregate metrics
        model_results['metrics'] = {
            'avg_coverage': np.mean(all_coverage_scores),
            'avg_quality': np.mean([np.mean(list(q.values())) for q in all_quality_scores]),
            'avg_consistency': np.mean(all_consistency_scores),
            'avg_response_time': np.mean(all_response_times),
            'avg_hallucination_score': np.mean(all_hallucination_scores),
            'std_coverage': np.std(all_coverage_scores),
            'std_quality': np.std([np.mean(list(q.values())) for q in all_quality_scores]),
            'std_consistency': np.std(all_consistency_scores),
            'std_response_time': np.std(all_response_times),
            'percentile_95_response_time': np.percentile(all_response_times, 95)
        }

        return model_results

    def run_evaluation(self, sample_size: int = 50):
        """Run complete evaluation pipeline"""
        print(f"Starting Perception Layer Evaluation")
        print(f"Dataset: {self.dataset_path}")
        print(f"Sample size: {sample_size} per category")

        # Load dataset
        samples = self.load_dataset(sample_size=sample_size)
        print(f"Total samples loaded: {len(samples)}")

        # Evaluate each model
        all_results = []
        for model_config in MODELS_TO_EVALUATE:
            model_results = self.evaluate_model(model_config, samples[:10])  # Limited for demo
            all_results.append(model_results)

            # Save individual model results
            self.save_model_results(model_results)

        # Comparative analysis
        comparison_df = self.create_comparison_table(all_results)
        self.save_comparison_results(comparison_df)

        # Identify best model
        best_model = self.identify_best_model(all_results)

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Best performing model: {best_model['name']}")
        print(f"Key strengths: {best_model['strengths']}")
        print(f"Areas for improvement: {best_model['improvements']}")

        return all_results, best_model

    def create_comparison_table(self, all_results: List[Dict]) -> pd.DataFrame:
        """Create comparison table of all models"""
        comparison_data = []

        for result in all_results:
            row = {
                'Model': result['model_name'],
                'Avg Coverage': f"{result['metrics']['avg_coverage']:.3f}",
                'Avg Quality': f"{result['metrics']['avg_quality']:.3f}",
                'Avg Consistency': f"{result['metrics']['avg_consistency']:.3f}",
                'Avg Response Time': f"{result['metrics']['avg_response_time']:.2f}s",
                'Hallucination Score': f"{result['metrics']['avg_hallucination_score']:.3f}",
                'Overall Score': 0  # Will calculate
            }

            # Calculate weighted overall score
            weights = {
                'coverage': 0.25,
                'quality': 0.20,
                'consistency': 0.25,
                'efficiency': 0.15,
                'hallucination': 0.15
            }

            efficiency_score = 1.0 if result['metrics']['avg_response_time'] < 1.0 else \
                              0.7 if result['metrics']['avg_response_time'] < 3.0 else 0.4

            overall = (
                weights['coverage'] * result['metrics']['avg_coverage'] +
                weights['quality'] * result['metrics']['avg_quality'] +
                weights['consistency'] * result['metrics']['avg_consistency'] +
                weights['efficiency'] * efficiency_score +
                weights['hallucination'] * result['metrics']['avg_hallucination_score']
            )

            row['Overall Score'] = f"{overall:.3f}"
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Overall Score', ascending=False)

        return df

    def identify_best_model(self, all_results: List[Dict]) -> Dict:
        """Identify the best model and its characteristics"""
        # Calculate overall scores
        scored_models = []

        for result in all_results:
            weights = {
                'coverage': 0.25,
                'quality': 0.20,
                'consistency': 0.25,
                'efficiency': 0.15,
                'hallucination': 0.15
            }

            efficiency_score = 1.0 if result['metrics']['avg_response_time'] < 1.0 else \
                              0.7 if result['metrics']['avg_response_time'] < 3.0 else 0.4

            overall = (
                weights['coverage'] * result['metrics']['avg_coverage'] +
                weights['quality'] * result['metrics']['avg_quality'] +
                weights['consistency'] * result['metrics']['avg_consistency'] +
                weights['efficiency'] * efficiency_score +
                weights['hallucination'] * result['metrics']['avg_hallucination_score']
            )

            scored_models.append({
                'name': result['model_name'],
                'score': overall,
                'metrics': result['metrics']
            })

        # Sort by score
        scored_models.sort(key=lambda x: x['score'], reverse=True)
        best = scored_models[0]

        # Analyze strengths and weaknesses
        strengths = []
        improvements = []

        metrics = best['metrics']

        if metrics['avg_coverage'] > 0.8:
            strengths.append("Excellent attribute coverage")
        elif metrics['avg_coverage'] < 0.6:
            improvements.append("Improve attribute extraction completeness")

        if metrics['avg_quality'] > 0.7:
            strengths.append("High-quality descriptions")
        elif metrics['avg_quality'] < 0.5:
            improvements.append("Enhance description quality and marketing appeal")

        if metrics['avg_consistency'] > 0.8:
            strengths.append("Strong semantic consistency")
        elif metrics['avg_consistency'] < 0.6:
            improvements.append("Better align descriptions with attributes")

        if metrics['avg_response_time'] < 1.0:
            strengths.append("Fast response time")
        elif metrics['avg_response_time'] > 3.0:
            improvements.append("Optimize for faster inference")

        if metrics['avg_hallucination_score'] > 0.8:
            strengths.append("Low hallucination rate")
        elif metrics['avg_hallucination_score'] < 0.6:
            improvements.append("Reduce hallucinations and improve accuracy")

        return {
            'name': best['name'],
            'overall_score': best['score'],
            'strengths': strengths,
            'improvements': improvements,
            'metrics': best['metrics']
        }

    def save_model_results(self, model_results: Dict):
        """Save individual model results"""
        filename = f"{self.output_dir}/{model_results['model_name'].replace('/', '_')}_results.json"
        with open(filename, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        print(f"Saved results to {filename}")

    def save_comparison_results(self, comparison_df: pd.DataFrame):
        """Save comparison results"""
        # Save as CSV
        csv_filename = f"{self.output_dir}/model_comparison.csv"
        comparison_df.to_csv(csv_filename, index=False)

        # Save as formatted text
        txt_filename = f"{self.output_dir}/model_comparison.txt"
        with open(txt_filename, 'w') as f:
            f.write("PERCEPTION LAYER MODEL COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write(f"Evaluation completed at: {datetime.now()}\n")

        print(f"Saved comparison to {csv_filename} and {txt_filename}")

# ===================== Main Execution =====================
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = PerceptionLayerEvaluator(
        dataset_path="dataset.json",
        output_dir="/home/claude/perception_evaluation_results"
    )

    # Run evaluation
    results, best_model = evaluator.run_evaluation(sample_size=5)  # Small sample for demo

    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(f"\nBest Model: {best_model['name']}")
    print(f"Overall Score: {best_model['overall_score']:.3f}")
    print(f"\nStrengths:")
    for strength in best_model['strengths']:
        print(f"  ✓ {strength}")
    print(f"\nAreas for Improvement:")
    for improvement in best_model['improvements']:
        print(f"  • {improvement}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Deploy the best model as primary perception layer")
    print("2. Train lightweight correction model for identified weaknesses")
    print("3. Implement prompt optimization based on evaluation results")
    print("4. Set up continuous monitoring for production performance")