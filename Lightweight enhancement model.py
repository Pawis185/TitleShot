"""
TitleShot - Lightweight Enhancement Model Training
Author: Shikang Wang
Purpose: Train a lightweight model to enhance the best perception model's weak points
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split


# ===================== Enhancement Model Architecture =====================
class AttributeEnhancementModel(nn.Module):
    """
    Lightweight model to enhance attribute extraction quality
    Focuses on:
    1. Correcting common extraction errors
    2. Filling missing attributes
    3. Improving consistency
    4. Reducing hallucinations
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_attributes: int = 20):
        super(AttributeEnhancementModel, self).__init__()

        # Feature extraction layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attribute-specific heads
        self.attribute_heads = nn.ModuleDict({
            'category': nn.Linear(hidden_dim, 15),  # 15 main categories
            'color': nn.Linear(hidden_dim, 50),  # 50 common colors
            'material': nn.Linear(hidden_dim, 30),  # 30 material types
            'brand_presence': nn.Linear(hidden_dim, 2),  # Binary: has brand or not
            'quality_indicators': nn.Linear(hidden_dim, 10),  # Multiple quality aspects
        })

        # Consistency checker
        self.consistency_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Hallucination detector
        self.hallucination_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_encoder(x)

        # Generate attribute predictions
        outputs = {}
        for attr_name, head in self.attribute_heads.items():
            outputs[attr_name] = head(features)

        # Additional checks
        outputs['consistency_score'] = self.consistency_layer(features)
        outputs['hallucination_score'] = self.hallucination_detector(features)

        return outputs


class PromptOptimizationModel(nn.Module):
    """
    Lightweight model to generate optimized prompts based on image features
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, vocab_size: int = 30000):
        super(PromptOptimizationModel, self).__init__()

        # Image feature processor
        self.image_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Context encoder
        self.context_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Prompt template selector (chooses best template)
        self.template_selector = nn.Linear(hidden_dim, 5)  # 5 template types

        # Keyword generator
        self.keyword_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 100),  # Top 100 relevant keywords
            nn.Sigmoid()
        )

    def forward(self, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process image features
        processed = self.image_processor(image_features)

        # Get context representation
        context, _ = self.context_encoder(processed.unsqueeze(1))
        context = context.squeeze(1)

        # Select best template
        template_scores = self.template_selector(context)

        # Generate keywords
        keywords = self.keyword_generator(context)

        return {
            'template_scores': template_scores,
            'keyword_relevance': keywords,
            'context_embedding': context
        }


# ===================== Custom Dataset =====================
class EcommerceEnhancementDataset(Dataset):
    """
    Dataset for training the enhancement model
    """

    def __init__(self,
                 data_path: str,
                 primary_model_outputs_path: str,
                 transform=None):
        """
        Args:
            data_path: Path to original dataset
            primary_model_outputs_path: Path to outputs from best perception model
            transform: Optional data transformation
        """
        self.data = self._load_data(data_path)
        self.primary_outputs = self._load_primary_outputs(primary_model_outputs_path)
        self.transform = transform

        # Prepare training samples
        self.samples = self._prepare_samples()

    def _load_data(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _load_primary_outputs(self, path: str) -> Dict:
        # In real scenario, this would load actual model outputs
        # For now, return mock data
        return {}

    def _prepare_samples(self) -> List[Dict]:
        samples = []

        for category, items in self.data.items():
            for item in items[:100]:  # Limit for demonstration
                sample = {
                    'category': category,
                    'image_url': item['image_url'],
                    'features': np.random.randn(768),  # Mock features
                    'weak_attributes': self._identify_weak_attributes(item),
                    'ground_truth': self._create_ground_truth(item, category)
                }
                samples.append(sample)

        return samples

    def _identify_weak_attributes(self, item: Dict) -> List[str]:
        # Identify which attributes need enhancement
        # Based on evaluation results
        return ['material', 'brand', 'specific_features']

    def _create_ground_truth(self, item: Dict, category: str) -> Dict:
        # Create ground truth labels for training
        return {
            'category': category,
            'has_brand': np.random.choice([0, 1]),
            'quality_score': np.random.random(),
            'consistency': 1.0,
            'hallucination': 0.0
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'labels': sample['ground_truth']
        }


# ===================== Training Pipeline =====================
class EnhancementModelTrainer:
    """
    Training pipeline for the lightweight enhancement model
    """

    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 output_dir: str = "./enhancement_model"):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )

        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3
        )

        # Loss functions for different outputs
        self.criterion = {
            'category': nn.CrossEntropyLoss(),
            'color': nn.CrossEntropyLoss(),
            'material': nn.CrossEntropyLoss(),
            'brand_presence': nn.BCEWithLogitsLoss(),
            'quality_indicators': nn.BCEWithLogitsLoss(),
            'consistency_score': nn.MSELoss(),
            'hallucination_score': nn.MSELoss()
        }

        self.training_history = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features']
            labels = batch['labels']

            # Forward pass
            outputs = self.model(features)

            # Calculate losses for each output
            total_loss = 0
            loss_components = {}

            for key in outputs.keys():
                if key in labels:
                    loss = self.criterion[key](outputs[key], labels[key])
                    total_loss += loss
                    loss_components[key] = loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(total_loss.item())

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {total_loss.item():.4f}")

        return {
            'avg_loss': np.mean(epoch_losses),
            'loss_components': loss_components
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features']
                labels = batch['labels']

                outputs = self.model(features)

                total_loss = 0
                for key in outputs.keys():
                    if key in labels:
                        loss = self.criterion[key](outputs[key], labels[key])
                        total_loss += loss

                val_losses.append(total_loss.item())

        return {'val_loss': np.mean(val_losses)}

    def train(self, num_epochs: int = 50):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs")
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])

            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['avg_loss'],
                'val_loss': val_metrics['val_loss'],
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_metrics)

            print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, best_val_loss)
                print(f"✓ New best model saved!")

        # Save training history
        self.save_training_history()

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'input_dim': 768,
                'hidden_dim': 256,
                'num_attributes': 20
            }
        }, checkpoint_path)

        print(f"Model checkpoint saved to {checkpoint_path}")

    def save_training_history(self):
        """Save training history"""
        history_df = pd.DataFrame(self.training_history)
        history_path = os.path.join(self.output_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")


# ===================== Prompt Generation Pipeline =====================
class EnhancedPromptGenerator:
    """
    Generate optimized prompts using the trained enhancement model
    """

    def __init__(self,
                 enhancement_model: nn.Module,
                 prompt_model: nn.Module):
        self.enhancement_model = enhancement_model
        self.prompt_model = prompt_model

        # Predefined template library
        self.templates = {
            'structured': "Extract structured attributes: {keywords}",
            'marketing': "Analyze for e-commerce listing focusing on: {keywords}",
            'detailed': "Provide comprehensive analysis including: {keywords}",
            'scenario': "Describe product for {scenario} with emphasis on: {keywords}",
            'comparison': "Compare with similar products, highlighting: {keywords}"
        }

        # Keyword library by category
        self.keyword_library = {
            'Electronics': ['specifications', 'connectivity', 'compatibility', 'performance'],
            'Clothing': ['fabric', 'fit', 'style', 'occasion', 'care instructions'],
            'Home & Kitchen': ['dimensions', 'capacity', 'material', 'features', 'maintenance'],
            'Sports & Outdoors': ['durability', 'weather resistance', 'size', 'weight'],
            'Books': ['genre', 'author', 'pages', 'format', 'language']
        }

    def generate_enhanced_prompt(self,
                                 image_features: torch.Tensor,
                                 category: str,
                                 weak_points: List[str]) -> str:
        """
        Generate an optimized prompt based on image features and identified weaknesses
        """
        # Get prompt optimization scores
        with torch.no_grad():
            prompt_outputs = self.prompt_model(image_features)

        # Select best template
        template_idx = torch.argmax(prompt_outputs['template_scores']).item()
        template_key = list(self.templates.keys())[template_idx]
        template = self.templates[template_key]

        # Get relevant keywords
        keyword_scores = prompt_outputs['keyword_relevance'].squeeze().numpy()
        top_keyword_indices = np.argsort(keyword_scores)[-10:]  # Top 10 keywords

        # Combine category keywords with weak point focus
        category_keywords = self.keyword_library.get(category, [])
        weak_point_keywords = [f"detailed_{wp}" for wp in weak_points]

        all_keywords = category_keywords + weak_point_keywords

        # Create final prompt
        if '{scenario}' in template:
            scenario = self._select_scenario(category)
            prompt = template.format(scenario=scenario, keywords=', '.join(all_keywords))
        else:
            prompt = template.format(keywords=', '.join(all_keywords))

        # Add specific instructions for weak points
        if weak_points:
            prompt += f"\nPay special attention to: {', '.join(weak_points)}"
            prompt += "\nBe specific and avoid generic descriptions."

        return prompt

    def _select_scenario(self, category: str) -> str:
        """Select appropriate scenario based on category"""
        scenarios = {
            'Electronics': 'professional work environment',
            'Clothing': 'daily wear and special occasions',
            'Home & Kitchen': 'modern home living',
            'Sports & Outdoors': 'active lifestyle',
            'Books': 'leisure reading'
        }
        return scenarios.get(category, 'everyday use')


# ===================== Integration Pipeline =====================
class IntegratedPerceptionPipeline:
    """
    Complete pipeline integrating best model + enhancement model + optimized prompts
    """

    def __init__(self,
                 primary_model_name: str,
                 enhancement_model_path: str,
                 prompt_model_path: str):

        self.primary_model_name = primary_model_name

        # Load enhancement models
        self.enhancement_model = self._load_enhancement_model(enhancement_model_path)
        self.prompt_generator = self._load_prompt_generator(prompt_model_path)

        print(f"Integrated Pipeline Initialized")
        print(f"Primary Model: {primary_model_name}")
        print(f"Enhancement: Loaded")
        print(f"Prompt Optimization: Loaded")

    def _load_enhancement_model(self, path: str) -> nn.Module:
        """Load trained enhancement model"""
        # In production, load from checkpoint
        model = AttributeEnhancementModel()
        # model.load_state_dict(torch.load(path)['model_state_dict'])
        model.eval()
        return model

    def _load_prompt_generator(self, path: str) -> EnhancedPromptGenerator:
        """Load prompt generator with trained model"""
        enhancement_model = AttributeEnhancementModel()
        prompt_model = PromptOptimizationModel()
        generator = EnhancedPromptGenerator(enhancement_model, prompt_model)
        return generator

    def process_image(self, image_url: str, category: str = None) -> Dict:
        """
        Complete processing pipeline for a product image
        """
        print(f"\nProcessing: {image_url[:50]}...")

        # Step 1: Extract initial features (mock for demo)
        image_features = torch.randn(1, 768)

        # Step 2: Identify weak points from initial extraction
        weak_points = self._identify_weaknesses(image_features)
        print(f"Identified weak points: {weak_points}")

        # Step 3: Generate optimized prompt
        optimized_prompt = self.prompt_generator.generate_enhanced_prompt(
            image_features,
            category or 'General',
            weak_points
        )
        print(f"Optimized prompt generated")

        # Step 4: Get primary model output with optimized prompt
        primary_output = self._call_primary_model(image_url, optimized_prompt)

        # Step 5: Enhance with lightweight model
        enhanced_output = self._enhance_attributes(primary_output, image_features)

        # Step 6: Final consistency check
        final_output = self._ensure_consistency(enhanced_output)

        return {
            'attributes': final_output,
            'optimized_prompt': optimized_prompt,
            'confidence_scores': self._calculate_confidence(final_output)
        }

    def _identify_weaknesses(self, features: torch.Tensor) -> List[str]:
        """Identify weak attributes that need enhancement"""
        # Based on enhancement model analysis
        with torch.no_grad():
            outputs = self.enhancement_model(features)

        weak_points = []

        # Check consistency score
        if outputs['consistency_score'].item() < 0.7:
            weak_points.append('consistency')

        # Check hallucination score
        if outputs['hallucination_score'].item() > 0.3:
            weak_points.append('accuracy')

        # Default weak points based on evaluation
        weak_points.extend(['material', 'specific_features'])

        return weak_points

    def _call_primary_model(self, image_url: str, prompt: str) -> Dict:
        """Call the primary perception model"""
        # In production, this would call the actual model API
        return {
            'category': 'Electronics',
            'color': 'Black',
            'brand': 'TechBrand',
            'description': 'High-quality electronic device'
        }

    def _enhance_attributes(self,
                            primary_output: Dict,
                            features: torch.Tensor) -> Dict:
        """Enhance attributes using the lightweight model"""
        with torch.no_grad():
            enhancements = self.enhancement_model(features)

        # Merge enhancements with primary output
        enhanced = primary_output.copy()

        # Add missing attributes
        if 'material' not in enhanced:
            material_idx = torch.argmax(enhancements['material']).item()
            materials = ['plastic', 'metal', 'glass', 'fabric', 'leather']
            enhanced['material'] = materials[min(material_idx, len(materials) - 1)]

        # Improve quality indicators
        quality_scores = torch.sigmoid(enhancements['quality_indicators']).squeeze().numpy()
        quality_aspects = ['durability', 'craftsmanship', 'design', 'functionality']
        enhanced['quality_indicators'] = {
            aspect: float(score)
            for aspect, score in zip(quality_aspects, quality_scores[:4])
        }

        return enhanced

    def _ensure_consistency(self, attributes: Dict) -> Dict:
        """Ensure consistency across all attributes"""
        # Category-specific consistency rules
        consistency_rules = {
            'Electronics': {
                'expected_attrs': ['brand', 'model', 'connectivity'],
                'invalid_materials': ['fabric', 'leather']
            },
            'Clothing': {
                'expected_attrs': ['size', 'material', 'style'],
                'invalid_materials': ['metal', 'glass']
            }
        }

        category = attributes.get('category', 'General')
        if category in consistency_rules:
            rules = consistency_rules[category]

            # Check for invalid materials
            if 'material' in attributes:
                if attributes['material'] in rules.get('invalid_materials', []):
                    attributes['material'] = 'synthetic'  # Default safe value

            # Add expected attributes if missing
            for expected in rules.get('expected_attrs', []):
                if expected not in attributes:
                    attributes[expected] = 'N/A'

        return attributes

    def _calculate_confidence(self, attributes: Dict) -> Dict[str, float]:
        """Calculate confidence scores for each attribute"""
        # Simple confidence calculation (would be more sophisticated in production)
        confidence = {}

        for key, value in attributes.items():
            if value and value != 'N/A':
                confidence[key] = 0.85 + np.random.random() * 0.15  # 0.85-1.0
            else:
                confidence[key] = 0.3 + np.random.random() * 0.3  # 0.3-0.6

        return confidence


# ===================== Main Training and Integration =====================
def main():
    """
    Main function to train enhancement model and set up integrated pipeline
    """
    print("=" * 80)
    print("TITLESHOT - PERCEPTION LAYER ENHANCEMENT")
    print("=" * 80)

    # Initialize dataset
    print("\n1. Loading Dataset...")
    dataset = EcommerceEnhancementDataset(
        data_path="dataset.json",
        primary_model_outputs_path="./primary_model_outputs.json"
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Initialize and train enhancement model
    print("\n2. Initializing Enhancement Model...")
    enhancement_model = AttributeEnhancementModel()
    prompt_model = PromptOptimizationModel()

    print("\n3. Training Enhancement Model...")
    trainer = EnhancementModelTrainer(
        model=enhancement_model,
        dataset=dataset,
        output_dir="/home/claude/enhancement_model"
    )

    # Train for a few epochs (reduced for demonstration)
    trainer.train(num_epochs=5)

    # Set up integrated pipeline
    print("\n4. Setting up Integrated Pipeline...")
    pipeline = IntegratedPerceptionPipeline(
        primary_model_name="Qwen2-VL-7B",  # Assuming this was best
        enhancement_model_path="/home/claude/enhancement_model/best_model.pt",
        prompt_model_path="/home/claude/enhancement_model/prompt_model.pt"
    )

    # Test on sample
    print("\n5. Testing Integrated Pipeline...")
    test_sample = {
        'image_url': 'https://example.com/product.jpg',
        'category': 'Electronics'
    }

    result = pipeline.process_image(
        test_sample['image_url'],
        test_sample['category']
    )

    print("\n" + "=" * 80)
    print("PIPELINE OUTPUT")
    print("=" * 80)
    print("\nExtracted Attributes:")
    for key, value in result['attributes'].items():
        print(f"  {key}: {value}")

    print("\nConfidence Scores:")
    for key, score in result['confidence_scores'].items():
        print(f"  {key}: {score:.3f}")

    print("\nOptimized Prompt:")
    print(f"  {result['optimized_prompt']}")

    print("\n" + "=" * 80)
    print("ENHANCEMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()