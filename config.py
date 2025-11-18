# -*- coding: utf-8 -*-
"""
config.py

全局配置文件：
- 商品类别列表（与 dataset.json 保持一致）
- 感知层模型内部名称列表
- 远程推理端点地址（可从环境变量覆盖）
"""

from typing import List
import os

# =========================
# 商品类别（感知层分类目标）
# =========================
CATEGORIES: List[str] = [
    "Electronics",
    "Home_and_Kitchen",
    "Books",
    "Clothing_Shoes_and_Jewelry",
    "Sports_and_Outdoors",
    "Movies_and_TV",
    "Automotive",
    "Tools_and_Home_Improvement",
    "Pet_Supplies",
    "Health_and_Personal_Care",
    "Toys_and_Games",
    "Office_Products",
]

# =========================
# 感知层候选模型（内部名称）
# =========================
PERCEPTION_MODELS: List[str] = [
    "qwen3_vl_32b",         # Qwen/Qwen3-VL-32B-Instruct (HF Endpoint, 文本接口)
    "internvl2_5_8b",       # InternVL2_5-8B (Friendli Dedicated)
    "minicpm_llama3_v_2_5", # MiniCPM-Llama3-V-2_5 (Friendli Dedicated)
    "blip2_opt_2_7b_nli",   # blip2-opt-2-7b-nli (HF Endpoint, 文本接口)
]

# =========================
# Hugging Face 端点（可被环境变量覆盖）
# =========================

# Qwen3-VL-32B-Instruct 的 HF Endpoint（不用 /v1/chat/completions，直接 POST 根地址）
HF_QWEN3_ENDPOINT = os.getenv(
    "HF_QWEN3_ENDPOINT",
    "https://bfhfmuskpdnp4ajz.us-east-1.aws.endpoints.huggingface.cloud",
)

# BLIP-2 HF Endpoint
HF_BLIP2_ENDPOINT = os.getenv(
    "HF_BLIP2_ENDPOINT",
    "https://onnunaok1tz3tbo2.us-east-1.aws.endpoints.huggingface.cloud",
)

# =========================
# Friendli Dedicated 基础 URL
# =========================

FRIENDLI_BASE_URL = "https://api.friendli.ai"
