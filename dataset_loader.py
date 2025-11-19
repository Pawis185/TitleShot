# dataset_loader.py
# 负责加载 dataset.json 并展开成样本列表

import json
from typing import List, Dict, Any


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    """
    从 dataset.json 读取数据，并展开成统一列表形式：
    [{
        "image_url": "...",
        "label": "Electronics"
     }, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples: List[Dict[str, Any]] = []
    for category, items in raw.items():
        for item in items:
            # item 里一般有: { "category": "...", "image_url": "..." }
            image_url = item["image_url"]
            samples.append(
                {
                    "image_url": image_url,
                    "label": category,
                }
            )
    return samples
