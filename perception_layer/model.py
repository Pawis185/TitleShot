"""Upper-level perception layer models for TitleShot.

This module implements reusable components for the perception layer with support for
multiple state-of-the-art vision-language models:
- BLIP-2: Baseline model with good balance
- LLaVA: Strong instruction-following capabilities
- Qwen2-VL: Best for Chinese e-commerce (recommended)
- InternVL2: Top-tier multimodal understanding
- MiniCPM-V: Lightweight solution for resource-constrained environments
- Florence-2: Fast structured output for baseline comparison

The :class:`PerceptionPipeline` orchestrates visual and textual processing to produce
unified outputs for the downstream generation layer.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


# Type aliases
AttributeDict = Dict[str, Any]


class ModelType(Enum):
    """Supported vision-language models for attribute extraction."""

    BLIP2_FLAN_T5_XL = "blip2-flan-t5-xl"
    BLIP2_OPT_2_7B = "blip2-opt-2.7b"
    LLAVA_1_5_7B = "llava-1.5-7b"
    QWEN2_VL_7B = "qwen2-vl-7b"
    INTERNVL2_8B = "internvl2-8b"
    MINICPM_V_2_5 = "minicpm-v-2.5"
    FLORENCE2_LARGE = "florence2-large"


# Model configuration mapping
MODEL_CONFIGS = {
    ModelType.BLIP2_FLAN_T5_XL: {
        "hf_name": "Salesforce/blip2-flan-t5-xl",
        "type": "blip2",
        "min_vram_gb": 16,
        "description": "Baseline BLIP-2 with good balance",
    },
    ModelType.BLIP2_OPT_2_7B: {
        "hf_name": "Salesforce/blip2-opt-2.7b",
        "type": "blip2",
        "min_vram_gb": 12,
        "description": "Lighter BLIP-2 variant",
    },
    ModelType.LLAVA_1_5_7B: {
        "hf_name": "llava-hf/llava-1.5-7b-hf",
        "type": "llava",
        "min_vram_gb": 16,
        "description": "Strong instruction-following",
    },
    ModelType.QWEN2_VL_7B: {
        "hf_name": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2",
        "min_vram_gb": 16,
        "description": "⭐ Best for Chinese e-commerce",
    },
    ModelType.INTERNVL2_8B: {
        "hf_name": "OpenGVLab/InternVL2-8B",
        "type": "internvl2",
        "min_vram_gb": 18,
        "description": "⭐ Top-tier multimodal understanding",
    },
    ModelType.MINICPM_V_2_5: {
        "hf_name": "openbmb/MiniCPM-Llama3-V-2_5",
        "type": "minicpm",
        "min_vram_gb": 8,
        "description": "💡 Lightweight, mobile-friendly",
    },
    ModelType.FLORENCE2_LARGE: {
        "hf_name": "microsoft/Florence-2-large",
        "type": "florence2",
        "min_vram_gb": 4,
        "description": "⚡ Fast structured output",
    },
}


@dataclass
class ImageQualityMetrics:
    """Metrics for assessing image quality."""

    width: int
    height: int
    mean_brightness: float
    is_too_dark: bool
    is_too_bright: bool
    is_low_resolution: bool

    @property
    def is_acceptable(self) -> bool:
        """Check if image meets minimum quality standards."""
        return not (self.is_too_dark or self.is_too_bright or self.is_low_resolution)


@dataclass
class AttributeExtractionResult:
    """Holds the raw text output of the vision-language model and parsed attributes."""

    raw_output: str
    attributes: AttributeDict = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    extraction_time: float = 0.0
    model_type: Optional[str] = None
    image_quality: Optional[ImageQualityMetrics] = None

    @property
    def has_all_required_attributes(self) -> bool:
        """Check if all essential attributes are extracted."""
        required = {"category", "color", "material", "style", "selling_points"}
        return all(attr in self.attributes for attr in required)


class VisualAttributeExtractor:
    """Extracts structured attributes from e-commerce images using vision-language models.

    Supports multiple models with automatic fallback and quality checks.

    Example::
        extractor = VisualAttributeExtractor(ModelType.QWEN2_VL_7B)
        result = extractor.extract("product.jpg")
        print(result.attributes)
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.BLIP2_FLAN_T5_XL,
        device: Optional[str] = None,
        check_image_quality: bool = True,
    ) -> None:
        """Initialize the visual attribute extractor.

        Args:
            model_type: Type of vision-language model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            check_image_quality: Whether to perform image quality checks
        """
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.check_quality = check_image_quality

        # Print model info
        print(f"🚀 Loading {model_type.value}: {self.config['description']}")
        print(f"📦 Model: {self.config['hf_name']}")
        print(f"💾 Min VRAM: {self.config['min_vram_gb']}GB")

        # Load model based on type
        self._load_model()

    def _load_model(self) -> None:
        """Load the appropriate model based on model_type."""
        model_family = self.config["type"]

        if model_family == "blip2":
            self._load_blip2()
        elif model_family == "llava":
            self._load_llava()
        elif model_family == "qwen2":
            self._load_qwen2()
        elif model_family == "internvl2":
            self._load_internvl2()
        elif model_family == "minicpm":
            self._load_minicpm()
        elif model_family == "florence2":
            self._load_florence2()
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

    def _load_blip2(self) -> None:
        """Load BLIP-2 model."""
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.config["hf_name"])
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.config["hf_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_llava(self) -> None:
        """Load LLaVA model."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.config["hf_name"])
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config["hf_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_qwen2(self) -> None:
        """Load Qwen2-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Qwen2-VL requires: pip install qwen-vl-utils transformers>=4.37.0"
            )

        self.processor = AutoProcessor.from_pretrained(self.config["hf_name"])
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config["hf_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_internvl2(self) -> None:
        """Load InternVL2 model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("InternVL2 requires: pip install timm einops")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_name"], trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config["hf_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_minicpm(self) -> None:
        """Load MiniCPM-V model."""
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["hf_name"], trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config["hf_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_florence2(self) -> None:
        """Load Florence-2 model."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.processor = AutoProcessor.from_pretrained(
            self.config["hf_name"], trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["hf_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _check_image_quality(self, image: Image.Image) -> ImageQualityMetrics:
        """Assess image quality for e-commerce attribute extraction.

        Args:
            image: PIL Image object

        Returns:
            ImageQualityMetrics with quality assessment
        """
        width, height = image.size
        img_array = np.array(image.convert("RGB"))
        mean_brightness = float(img_array.mean())

        metrics = ImageQualityMetrics(
            width=width,
            height=height,
            mean_brightness=mean_brightness,
            is_too_dark=mean_brightness < 20,
            is_too_bright=mean_brightness > 235,
            is_low_resolution=(width < 224 or height < 224),
        )

        if not metrics.is_acceptable:
            warnings.warn(
                f"⚠️  Image quality issues detected:\n"
                f"   - Size: {width}x{height} (min 224x224)\n"
                f"   - Brightness: {mean_brightness:.1f} (ideal: 50-200)\n"
                f"   - Too dark: {metrics.is_too_dark}\n"
                f"   - Too bright: {metrics.is_too_bright}"
            )

        return metrics

    def _get_prompt(self) -> str:
        """Generate model-specific prompt for attribute extraction."""
        base_prompt = (
            "You are an expert e-commerce product analyst. "
            "Analyze this product image and extract key marketing attributes.\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            "{\n"
            '  "category": "main product category (e.g., clothing, electronics)",\n'
            '  "color": "primary color",\n'
            '  "material": "main material or fabric",\n'
            '  "style": "design style (e.g., casual, formal, modern)",\n'
            '  "selling_points": ["key feature 1", "key feature 2"]\n'
            "}\n\n"
            "Focus on attributes that would attract customers and improve search visibility."
        )

        # Model-specific adjustments
        if self.config["type"] in ["qwen2", "internvl2"]:
            # These models are better at Chinese
            base_prompt += "\n\n如果是中文商品，请用中文回答。"

        return base_prompt

    def _extract_blip2(self, image: Image.Image, prompt: str) -> str:
        """Extract attributes using BLIP-2."""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output.strip()

    def _extract_llava(self, image: Image.Image, prompt: str) -> str:
        """Extract attributes using LLaVA."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
            )
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output.strip()

    def _extract_qwen2(self, image: Image.Image, prompt: str) -> str:
        """Extract attributes using Qwen2-VL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
            )
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output.strip()

    def _extract_internvl2(self, image: Image.Image, prompt: str) -> str:
        """Extract attributes using InternVL2."""
        pixel_values = self.model.load_image(image).to(self.device)
        generation_config = dict(
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
        )

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )
        return response.strip()

    def _extract_minicpm(self, image: Image.Image, prompt: str) -> str:
        """Extract attributes using MiniCPM-V."""
        msgs = [{"role": "user", "content": prompt}]

        with torch.no_grad():
            response = self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7,
            )
        return response.strip()

    def _extract_florence2(self, image: Image.Image, prompt: str) -> str:
        """Extract attributes using Florence-2."""
        # Florence-2 uses task-specific prompts
        task_prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
            )

        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Post-process Florence output to match expected format
        return self._florence_to_json(output, prompt)

    def _florence_to_json(self, florence_output: str, original_prompt: str) -> str:
        """Convert Florence-2 caption to JSON format."""
        # Florence gives detailed captions, we need to structure them
        # This is a simplified conversion - in production you'd use more sophisticated NLP
        try:
            # Try to extract key info from the caption
            caption = florence_output.lower()

            # Simple heuristic extraction
            attributes = {
                "category": self._extract_category_from_text(caption),
                "color": self._extract_color_from_text(caption),
                "material": self._extract_material_from_text(caption),
                "style": self._extract_style_from_text(caption),
                "selling_points": [florence_output[:100], "detailed product image"],
            }

            return json.dumps(attributes)
        except Exception:
            return florence_output

    @staticmethod
    def _extract_category_from_text(text: str) -> str:
        """Extract category from text using keywords."""
        categories = {
            "clothing": ["shirt", "dress", "pants", "jacket", "coat", "sweater"],
            "electronics": ["phone", "laptop", "camera", "headphone", "speaker"],
            "furniture": ["chair", "table", "sofa", "desk", "bed"],
            "accessories": ["watch", "bag", "belt", "wallet", "jewelry"],
        }

        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category
        return "general"

    @staticmethod
    def _extract_color_from_text(text: str) -> str:
        """Extract color from text using keywords."""
        colors = ["red", "blue", "green", "yellow", "black", "white", "gray", "pink", "purple"]
        for color in colors:
            if color in text:
                return color
        return "multicolor"

    @staticmethod
    def _extract_material_from_text(text: str) -> str:
        """Extract material from text using keywords."""
        materials = ["cotton", "leather", "metal", "plastic", "wood", "silk", "polyester"]
        for material in materials:
            if material in text:
                return material
        return "mixed materials"

    @staticmethod
    def _extract_style_from_text(text: str) -> str:
        """Extract style from text using keywords."""
        styles = ["casual", "formal", "modern", "vintage", "sporty", "elegant"]
        for style in styles:
            if style in text:
                return style
        return "versatile"

    def extract(self, image_path: str | Path) -> AttributeExtractionResult:
        """Extract attributes from a product image.

        Args:
            image_path: Path to the product image

        Returns:
            AttributeExtractionResult with extracted attributes and metadata

        Raises:
            ValueError: If image quality is too poor
        """
        start_time = time.time()

        # Load and check image
        image = Image.open(image_path).convert("RGB")
        quality_metrics = None

        if self.check_quality:
            quality_metrics = self._check_image_quality(image)
            if not quality_metrics.is_acceptable:
                if quality_metrics.is_low_resolution:
                    raise ValueError(
                        f"Image resolution too low: {quality_metrics.width}x{quality_metrics.height}. "
                        f"Minimum required: 224x224"
                    )

        # Get prompt and extract
        prompt = self._get_prompt()
        model_family = self.config["type"]

        try:
            if model_family == "blip2":
                raw_output = self._extract_blip2(image, prompt)
            elif model_family == "llava":
                raw_output = self._extract_llava(image, prompt)
            elif model_family == "qwen2":
                raw_output = self._extract_qwen2(image, prompt)
            elif model_family == "internvl2":
                raw_output = self._extract_internvl2(image, prompt)
            elif model_family == "minicpm":
                raw_output = self._extract_minicpm(image, prompt)
            elif model_family == "florence2":
                raw_output = self._extract_florence2(image, prompt)
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}") from e

        # Parse attributes
        attributes = self._parse_attributes(raw_output)

        extraction_time = time.time() - start_time

        return AttributeExtractionResult(
            raw_output=raw_output,
            attributes=attributes,
            extraction_time=extraction_time,
            model_type=self.model_type.value,
            image_quality=quality_metrics,
        )

    @staticmethod
    def _parse_attributes(text: str) -> AttributeDict:
        """Parse JSON-like model output into a dictionary with robust fallback."""

        # Try direct JSON parsing first
        try:
            # Clean common formatting issues
            cleaned = text.strip()
            # Remove markdown code blocks
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            # Remove trailing commas before closing braces
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: heuristic parsing
        result: AttributeDict = {}

        # Extract key-value pairs
        patterns = {
            "category": r'["\']?category["\']?\s*:\s*["\']([^"\']+)["\']',
            "color": r'["\']?color["\']?\s*:\s*["\']([^"\']+)["\']',
            "material": r'["\']?material["\']?\s*:\s*["\']([^"\']+)["\']',
            "style": r'["\']?style["\']?\s*:\s*["\']([^"\']+)["\']',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()

        # Extract selling points (array)
        selling_points_match = re.search(
            r'["\']?selling_points["\']?\s*:\s*\[(.*?)\]',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if selling_points_match:
            points_str = selling_points_match.group(1)
            points = re.findall(r'["\']([^"\']+)["\']', points_str)
            result["selling_points"] = points[:2] if points else []

        return result


class TextAttributeEncoder:
    """Encodes textual artifacts (titles, bullet points, reviews) into embeddings.

    Example::
        encoder = TextAttributeEncoder()
        embedding = encoder.encode("Soft cashmere scarf for winter")
        keywords = encoder.extract_keywords("Soft cashmere scarf for winter")
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_multilingual: bool = False,
    ) -> None:
        """Initialize the text encoder.

        Args:
            model_name: Name of the sentence transformer model
            use_multilingual: If True, use a multilingual model for Chinese support
        """
        if use_multilingual:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print("🌏 Using multilingual text encoder for Chinese support")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        """Return a dense embedding for downstream retrieval or fusion.

        Args:
            text: Input text to encode

        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract salient keywords using frequency-based heuristic.

        Args:
            text: Input text
            top_k: Number of keywords to extract

        Returns:
            List of top keywords
        """
        # Tokenize
        tokens = re.findall(r"[A-Za-z0-9#'+-]+", text.lower())

        # Expanded stop words for e-commerce
        stop_words = {
            "the", "and", "with", "for", "this", "that", "from", "your",
            "you", "are", "will", "has", "have", "been", "can", "all",
            "more", "about", "into", "through", "than", "other", "some",
            "our", "their", "also", "very", "well", "such", "get", "make",
        }

        filtered = [token for token in tokens if token not in stop_words and len(token) > 2]
        most_common = Counter(filtered).most_common(top_k)
        return [token for token, _ in most_common]


@dataclass
class PerceptionOutput:
    """Unified payload returned by :class:`PerceptionPipeline`.

    This structure is designed to be consumed by the generation layer (Yiyang Wang's module).
    """

    image_attributes: AttributeExtractionResult
    text_embedding: Optional[List[float]] = None
    keywords: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization and downstream consumption."""
        return {
            "image_attributes": self.image_attributes.attributes,
            "raw_image_output": self.image_attributes.raw_output,
            "extraction_time": self.image_attributes.extraction_time,
            "model_type": self.image_attributes.model_type,
            "has_all_attributes": self.image_attributes.has_all_required_attributes,
            "text_embedding": self.text_embedding,
            "keywords": self.keywords,
            "image_quality": {
                "width": self.image_attributes.image_quality.width,
                "height": self.image_attributes.image_quality.height,
                "mean_brightness": self.image_attributes.image_quality.mean_brightness,
                "is_acceptable": self.image_attributes.image_quality.is_acceptable,
            } if self.image_attributes.image_quality else None,
        }

    def to_generation_input(self) -> str:
        """Format output as a structured prompt for the generation layer.

        Returns:
            Formatted string suitable for LLM input
        """
        attrs = self.image_attributes.attributes

        prompt = "Product Attributes:\n"
        prompt += f"- Category: {attrs.get('category', 'N/A')}\n"
        prompt += f"- Color: {attrs.get('color', 'N/A')}\n"
        prompt += f"- Material: {attrs.get('material', 'N/A')}\n"
        prompt += f"- Style: {attrs.get('style', 'N/A')}\n"

        if "selling_points" in attrs:
            prompt += f"- Selling Points: {', '.join(attrs['selling_points'])}\n"

        if self.keywords:
            prompt += f"- Keywords: {', '.join(self.keywords)}\n"

        return prompt


class PerceptionPipeline:
    """High-level orchestration for the perception layer.

    Combines visual attribute extraction and text encoding into a unified pipeline.
    Supports multiple models for comparison and ensemble methods.

    Example::
        # Single model
        pipeline = PerceptionPipeline(model_type=ModelType.QWEN2_VL_7B)
        result = pipeline.run("product.jpg", text_hint="Soft cashmere scarf")
        print(result.to_dict())

        # Model comparison
        results = pipeline.compare_models("product.jpg", [
            ModelType.QWEN2_VL_7B,
            ModelType.FLORENCE2_LARGE
        ])
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.BLIP2_FLAN_T5_XL,
        visual_extractor: Optional[VisualAttributeExtractor] = None,
        text_encoder: Optional[TextAttributeEncoder] = None,
        device: Optional[str] = None,
        use_multilingual: bool = False,
    ) -> None:
        """Initialize the perception pipeline.

        Args:
            model_type: Type of vision-language model to use
            visual_extractor: Custom visual extractor (overrides model_type if provided)
            text_encoder: Custom text encoder
            device: Device to run models on
            use_multilingual: Whether to use multilingual text encoder
        """
        self.model_type = model_type
        self.device = device

        if visual_extractor is not None:
            self.visual_extractor = visual_extractor
        else:
            self.visual_extractor = VisualAttributeExtractor(
                model_type=model_type,
                device=device,
            )

        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            self.text_encoder = TextAttributeEncoder(use_multilingual=use_multilingual)

    def run(
        self,
        image_path: str | Path,
        text_hint: Optional[str] = None
    ) -> PerceptionOutput:
        """Run the complete perception pipeline on a single input.

        Args:
            image_path: Path to the product image
            text_hint: Optional textual hint (title, description, etc.)

        Returns:
            PerceptionOutput containing all extracted information
        """
        # Extract visual attributes
        image_attributes = self.visual_extractor.extract(image_path)

        # Process text if provided
        text_embedding: Optional[List[float]] = None
        keywords: Optional[List[str]] = None

        if text_hint:
            text_embedding = self.text_encoder.encode(text_hint)
            keywords = self.text_encoder.extract_keywords(text_hint)

        return PerceptionOutput(
            image_attributes=image_attributes,
            text_embedding=text_embedding,
            keywords=keywords,
        )

    def batch_run(
        self,
        image_paths: Iterable[str | Path],
        text_hints: Optional[Iterable[Optional[str]]] = None,
        show_progress: bool = True,
    ) -> List[PerceptionOutput]:
        """Run the pipeline across a batch of inputs.

        Args:
            image_paths: List of image paths
            text_hints: Optional list of text hints
            show_progress: Whether to show progress bar

        Returns:
            List of PerceptionOutput objects
        """
        text_hint_list: List[Optional[str]] = list(text_hints) if text_hints is not None else []
        outputs: List[PerceptionOutput] = []

        image_list = list(image_paths)
        total = len(image_list)

        for idx, image_path in enumerate(image_list):
            if show_progress:
                print(f"Processing {idx + 1}/{total}: {Path(image_path).name}")

            text_hint = text_hint_list[idx] if idx < len(text_hint_list) else None

            try:
                output = self.run(image_path, text_hint=text_hint)
                outputs.append(output)
            except Exception as e:
                print(f"⚠️  Failed to process {image_path}: {str(e)}")
                # Create empty result for failed cases
                empty_result = AttributeExtractionResult(
                    raw_output=f"Error: {str(e)}",
                    attributes={},
                    extraction_time=0.0,
                    model_type=self.model_type.value,
                )
                outputs.append(PerceptionOutput(image_attributes=empty_result))

        return outputs

    def compare_models(
        self,
        image_path: str | Path,
        model_types: List[ModelType],
        text_hint: Optional[str] = None,
    ) -> Dict[str, PerceptionOutput]:
        """Compare multiple models on the same image.

        Args:
            image_path: Path to the product image
            model_types: List of model types to compare
            text_hint: Optional textual hint

        Returns:
            Dictionary mapping model names to their outputs
        """
        results = {}

        print(f"\n🔬 Comparing {len(model_types)} models on {Path(image_path).name}")
        print("=" * 70)

        for model_type in model_types:
            print(f"\n📊 Testing {model_type.value}...")

            try:
                # Create new pipeline with this model
                pipeline = PerceptionPipeline(
                    model_type=model_type,
                    device=self.device,
                )

                # Run extraction
                result = pipeline.run(image_path, text_hint=text_hint)
                results[model_type.value] = result

                # Print summary
                print(f"   ✅ Success in {result.image_attributes.extraction_time:.2f}s")
                print(f"   Attributes: {len(result.image_attributes.attributes)} extracted")

            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                results[model_type.value] = None

        print("\n" + "=" * 70)
        return results

    def benchmark_models(
        self,
        image_paths: List[str | Path],
        model_types: List[ModelType],
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark multiple models on a dataset.

        Args:
            image_paths: List of test images
            model_types: List of models to benchmark

        Returns:
            Dictionary with benchmark results per model
        """
        results = {}

        print(f"\n🏁 Benchmarking {len(model_types)} models on {len(image_paths)} images")
        print("=" * 70)

        for model_type in model_types:
            print(f"\n📊 Benchmarking {model_type.value}...")

            pipeline = PerceptionPipeline(model_type=model_type, device=self.device)

            total_time = 0.0
            successful = 0
            complete_attrs = 0

            for idx, image_path in enumerate(image_paths):
                try:
                    result = pipeline.run(image_path)
                    total_time += result.image_attributes.extraction_time
                    successful += 1

                    if result.image_attributes.has_all_required_attributes:
                        complete_attrs += 1

                except Exception:
                    pass

            # Calculate metrics
            avg_time = total_time / successful if successful > 0 else 0
            success_rate = successful / len(image_paths) * 100
            completeness_rate = complete_attrs / successful * 100 if successful > 0 else 0

            results[model_type.value] = {
                "avg_extraction_time": avg_time,
                "success_rate": success_rate,
                "attribute_completeness": completeness_rate,
                "total_processed": successful,
                "total_failed": len(image_paths) - successful,
            }

            print(f"   ⏱️  Avg time: {avg_time:.2f}s")
            print(f"   ✅ Success rate: {success_rate:.1f}%")
            print(f"   📋 Completeness: {completeness_rate:.1f}%")

        print("\n" + "=" * 70)
        return results

    @staticmethod
    def print_model_recommendations() -> None:
        """Print model recommendations based on different scenarios."""
        print("\n" + "=" * 70)
        print("🎯 MODEL RECOMMENDATIONS FOR E-COMMERCE")
        print("=" * 70)

        recommendations = [
            {
                "scenario": "Chinese E-commerce (Taobao, JD, Pinduoduo)",
                "model": "Qwen2-VL-7B",
                "reason": "Best Chinese language support, optimized for Asian e-commerce",
            },
            {
                "scenario": "International E-commerce (Amazon, eBay)",
                "model": "InternVL2-8B",
                "reason": "Top-tier multimodal understanding, excellent for detailed products",
            },
            {
                "scenario": "High-volume processing (>10k images/day)",
                "model": "Florence-2-Large",
                "reason": "Fastest inference, structured output, 4GB VRAM only",
            },
            {
                "scenario": "Mobile/Edge deployment",
                "model": "MiniCPM-V-2.5",
                "reason": "Lightweight (2.5B params), can run on CPU",
            },
            {
                "scenario": "Fashion & Apparel",
                "model": "InternVL2-8B or Qwen2-VL-7B",
                "reason": "Excellent at fine-grained visual details (fabric, texture)",
            },
            {
                "scenario": "Electronics & Tech products",
                "model": "LLaVA-1.5-7B or BLIP-2",
                "reason": "Good at technical specifications and product features",
            },
            {
                "scenario": "Budget-constrained / Research baseline",
                "model": "BLIP-2-OPT-2.7B",
                "reason": "Good balance, well-documented, widely used",
            },
        ]

        for rec in recommendations:
            print(f"\n📌 {rec['scenario']}")
            print(f"   🥇 Recommended: {rec['model']}")
            print(f"   💡 Reason: {rec['reason']}")

        print("\n" + "=" * 70)
        print("💾 VRAM Requirements:")
        for model_type, config in MODEL_CONFIGS.items():
            print(f"   {model_type.value:20s} → {config['min_vram_gb']:2d}GB")
        print("=" * 70 + "\n")


# Convenience function for quick testing
def quick_extract(
    image_path: str | Path,
    model_type: ModelType = ModelType.BLIP2_FLAN_T5_XL,
    text_hint: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Quick extraction function for testing.

    Args:
        image_path: Path to product image
        model_type: Model to use
        text_hint: Optional text hint
        verbose: Whether to print results

    Returns:
        Dictionary with extraction results
    """
    pipeline = PerceptionPipeline(model_type=model_type)
    result = pipeline.run(image_path, text_hint=text_hint)
    output_dict = result.to_dict()

    if verbose:
        print("\n" + "=" * 70)
        print("📦 EXTRACTION RESULTS")
        print("=" * 70)
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        print("=" * 70 + "\n")

    return output_dict