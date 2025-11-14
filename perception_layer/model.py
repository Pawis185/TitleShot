"""Upper-level perception layer models for TitleShot.

This module implements reusable components for the perception layer described in the
project proposal.  It focuses on two key capabilities:

* Visual attribute extraction from raw product images using BLIP-2.
* Textual feature encoding using a light-weight sentence transformer.

The :class:`PerceptionPipeline` class orchestrates these components and produces a unified
output that can be consumed by the downstream generation layer.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Blip2ForConditionalGeneration


AttributeDict = Dict[str, str]


@dataclass
class AttributeExtractionResult:
    """Holds the raw text output of the vision-language model and parsed attributes."""

    raw_output: str
    attributes: AttributeDict = field(default_factory=dict)


class VisualAttributeExtractor:
    """Extracts structured attributes from e-commerce images using BLIP-2.

    The class defaults to ``Salesforce/blip2-flan-t5-xl`` which offers an excellent trade-off
    between accuracy and latency for attribute extraction tasks without requiring dedicated
    GPUs.  The extractor prompts the model to respond with JSON, making it easier for the
    downstream pipeline to consume.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_path: str | Path) -> AttributeExtractionResult:
        image = Image.open(image_path).convert("RGB")
        prompt = (
            "You are an e-commerce merchandiser. "
            "List the product category, color, material, style, and two selling points "
            "for the product in the image. Respond as compact JSON with the keys: "
            "category, color, material, style, selling_points."
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        attributes = self._parse_attributes(output_text)
        return AttributeExtractionResult(raw_output=output_text, attributes=attributes)

    @staticmethod
    def _parse_attributes(text: str) -> AttributeDict:
        """Parse JSON-like model output into a dictionary."""

        try:
            cleaned = text.replace("\n", " ").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback heuristic parsing
        result: AttributeDict = {}
        patterns = {
            "category": r"category\s*[:\-]\s*(?P<value>[^,]+)",
            "color": r"color\s*[:\-]\s*(?P<value>[^,]+)",
            "material": r"material\s*[:\-]\s*(?P<value>[^,]+)",
            "style": r"style\s*[:\-]\s*(?P<value>[^,]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                result[key] = match.group("value").strip()
        selling_points_match = re.search(
            r"selling points?\s*[:\-]\s*(?P<value>.+)", text, flags=re.IGNORECASE
        )
        if selling_points_match:
            value = selling_points_match.group("value")
            parts = [part.strip(" .") for part in re.split(r"[,;]\s*", value) if part.strip()]
            result["selling_points"] = parts[:2]
        return result


class TextAttributeEncoder:
    """Encodes textual artefacts (titles, bullet points, reviews) into embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        """Return a dense embedding for downstream retrieval or fusion."""

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract salient keywords using a simple frequency-based heuristic."""

        tokens = re.findall(r"[A-Za-z0-9#'+-]+", text.lower())
        stop_words = {
            "the",
            "and",
            "with",
            "for",
            "this",
            "that",
            "from",
            "your",
            "you",
            "are",
            "will",
        }
        filtered = [token for token in tokens if token not in stop_words]
        most_common = Counter(filtered).most_common(top_k)
        return [token for token, _ in most_common]


@dataclass
class PerceptionOutput:
    """Unified payload returned by :class:`PerceptionPipeline`."""

    image_attributes: AttributeExtractionResult
    text_embedding: Optional[List[float]] = None
    keywords: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_attributes": self.image_attributes.attributes,
            "raw_image_output": self.image_attributes.raw_output,
            "text_embedding": self.text_embedding,
            "keywords": self.keywords,
        }


class PerceptionPipeline:
    """High-level orchestration for the perception layer.

    Example::

        pipeline = PerceptionPipeline()
        result = pipeline.run("sample.jpg", text_hint="Soft cashmere scarf")
        print(result.to_dict())
    """

    def __init__(
        self,
        visual_extractor: Optional[VisualAttributeExtractor] = None,
        text_encoder: Optional[TextAttributeEncoder] = None,
    ) -> None:
        self.visual_extractor = visual_extractor or VisualAttributeExtractor()
        self.text_encoder = text_encoder or TextAttributeEncoder()

    def run(self, image_path: str | Path, text_hint: Optional[str] = None) -> PerceptionOutput:
        image_attributes = self.visual_extractor.extract(image_path)
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
    ) -> List[PerceptionOutput]:
        """Run the pipeline across a batch of inputs."""

        text_hint_list: List[Optional[str]] = list(text_hints) if text_hints is not None else []
        outputs: List[PerceptionOutput] = []
        for idx, image_path in enumerate(image_paths):
            text_hint = text_hint_list[idx] if idx < len(text_hint_list) else None
            outputs.append(self.run(image_path, text_hint=text_hint))
        return outputs