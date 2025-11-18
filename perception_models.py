# -*- coding: utf-8 -*-
"""
perception_models.py

封装对 4 个感知层模型的调用：
- Qwen3-VL-32B-Instruct（HF Endpoint，文本接口）
- InternVL2_5-8B（Friendli Dedicated，多模态）
- MiniCPM-Llama3-V-2_5（Friendli Dedicated，多模态）
- BLIP-2-OPT-2.7B-NLI（HF Endpoint，文本接口）

对外主函数：
    classify_image(model_name, image_url) -> (pred_label, latency)

说明：
    - pred_label 是归一化后的标准类别（在 config.CATEGORIES 中）或 "__UNKNOWN__"
"""

import os
import time
from typing import List, Tuple, Optional

from config import (
    CATEGORIES,
    PERCEPTION_MODELS,
)
from qwen3_client import call_qwen3_vl_endpoint
from blip2_client import call_blip2_endpoint
from friendli_client import call_friendli_chat_completion


# =========================
# 工具函数：类别归一化
# =========================

def _normalize_category(raw_text: str, categories: List[str]) -> Optional[str]:
    """
    将模型原始输出文本尽量映射到标准类别标签上。

    步骤：
    1. 去掉常见前缀（如“类别：”“Category:”等）和标点；
    2. 做精确匹配（区分大小写）；
    3. 做常见变体匹配（下划线 -> 空格等）；
    4. 做包含关系匹配；
    5. 失败则返回 None。
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # 去前缀
    prefixes = [
        "类别：", "类别:", "类：", "类:",
        "Category:", "category:",
        "类别为", "属于",
        "label:", "Label:",
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    # 去句号
    text = text.replace("。", "").replace(".", "").strip()

    # 完全匹配
    if text in categories:
        return text

    lower_text = text.lower()

    def gen_variants(cat: str) -> List[str]:
        base = cat
        base_lower = cat.lower()
        with_space = cat.replace("_", " ")
        with_space_lower = with_space.lower()
        with_amp = with_space.replace(" and ", " & ")
        with_amp_lower = with_amp.lower()
        return [
            base,
            base_lower,
            with_space,
            with_space_lower,
            with_amp,
            with_amp_lower,
        ]

    # 变体精确匹配
    for cat in categories:
        variants = gen_variants(cat)
        if text in variants or lower_text in variants:
            return cat

    # 包含关系匹配
    for cat in categories:
        variants = gen_variants(cat)
        for v in variants:
            if v and v.lower() in lower_text:
                return cat

    return None


# =========================
# Qwen3：通过文本接口分类（软多模态）
# =========================

def _classify_with_qwen3(image_url: str) -> str:
    """
    使用 Qwen3-VL-32B 的 HF Endpoint 对图片进行商品类别分类。
    这里是“软多模态”：把 image_url 写进 prompt，让模型依据图片链接和类别含义输出一个类别名。
    """
    labels_str = ", ".join(CATEGORIES)

    prompt = f"""
你是一个电商平台的多模态商品分类助手。
现在给你一张商品图片（通过 URL 提供）以及候选类别列表，请你判断该商品最适合属于哪一个类别。

- 图片 URL: {image_url}
- 候选类别（英文）: {labels_str}

要求：
1. 仔细根据图片内容和类别含义进行判断；
2. 最终只输出一个类别的英文名称，必须是候选列表中的一个；
3. 不要输出任何多余的解释或文字。

请直接给出类别名称：
"""

    raw = call_qwen3_vl_endpoint(prompt, max_new_tokens=16, temperature=0.0)
    # 清洗输出（只取首行，去引号）
    pred = raw.strip().split("\n")[0].strip()
    pred = pred.replace('"', "").replace("'", "")
    norm = _normalize_category(pred, CATEGORIES)
    return norm if norm is not None else "__UNKNOWN__"


# =========================
# BLIP-2：通过文本接口分类（软多模态）
# =========================

def _classify_with_blip2(image_url: str) -> str:
    """
    使用 BLIP-2 HF Endpoint 对图片进行商品类别分类（同样通过 prompt + URL 的方式）。
    """
    labels_str = ", ".join(CATEGORIES)

    prompt = f"""
You are an e-commerce product image classifier.

You will be given a product image via a URL, and a list of candidate categories.
Your task is to decide which single category best matches the product in the image.

- Image URL: {image_url}
- Candidate categories (English): {labels_str}

Requirements:
1. Output exactly ONE category.
2. The category MUST be one of the candidate categories above.
3. Do NOT output any explanation or extra text.

Just output the category name:
"""

    raw = call_blip2_endpoint(prompt, max_new_tokens=16, temperature=0.0)
    pred = raw.strip().split("\n")[0].strip()
    pred = pred.replace('"', "").replace("'", "")
    norm = _normalize_category(pred, CATEGORIES)
    return norm if norm is not None else "__UNKNOWN__"


# =========================
# InternVL / MiniCPM：Friendli 多模态 chat.completions
# =========================

def _classify_with_friendli_vlm(model_env_key: str, image_url: str) -> str:
    """
    使用 Friendli Dedicated 多模态模型（InternVL2.5 / MiniCPM）进行图片分类。

    参数：
        model_env_key : 读取 model_id 的环境变量名
                        - FRIENDLI_INTERNVL_MODEL_ID
                        - FRIENDLI_MINICPM_MODEL_ID
        image_url     : 商品图片 URL
    """
    model_id = os.getenv(model_env_key)
    if not model_id:
        raise RuntimeError(f"未设置环境变量 {model_env_key}，无法调用 Friendli 模型。")

    labels_str = ", ".join(CATEGORIES)

    system_prompt = (
        "你是一个电商平台的多模态商品分类模型，输入是一张商品图片以及候选类别列表，"
        "输出是一个英文类别标签，必须从候选列表中选择。"
    )

    user_text = (
        "下面是一张商品图片，请根据图片内容，从候选类别中选出最合适的一类。\n\n"
        f"图片 URL: {image_url}\n"
        f"候选类别（英文）: {labels_str}\n\n"
        "只输出一个类别的英文名称，不要输出其他文字。"
    )

    # 多模态消息结构：text + image_url
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    raw = call_friendli_chat_completion(
        model_id=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=16,
    )

    pred = raw.strip().split("\n")[0].strip()
    pred = pred.replace('"', "").replace("'", "")
    norm = _normalize_category(pred, CATEGORIES)
    return norm if norm is not None else "__UNKNOWN__"


# =========================
# 对外统一接口
# =========================

def classify_image(model_name: str, image_url: str) -> Tuple[str, float]:
    """
    统一分类函数。

    参数：
        model_name : 感知层模型内部名称（见 config.PERCEPTION_MODELS）
        image_url  : 商品图片 URL

    返回：
        (pred_label, latency)
        - pred_label : 预测类别（或 "__UNKNOWN__"）
        - latency    : 推理耗时（秒）
    """
    if model_name not in PERCEPTION_MODELS:
        raise ValueError(f"未知模型名称: {model_name}")

    start = time.time()
    try:
        if model_name == "qwen3_vl_32b":
            pred = _classify_with_qwen3(image_url)

        elif model_name == "blip2_opt_2_7b_nli":
            pred = _classify_with_blip2(image_url)

        elif model_name == "internvl2_5_8b":
            pred = _classify_with_friendli_vlm(
                model_env_key="FRIENDLI_INTERNVL_MODEL_ID",
                image_url=image_url,
            )

        elif model_name == "minicpm_llama3_v_2_5":
            pred = _classify_with_friendli_vlm(
                model_env_key="FRIENDLI_MINICPM_MODEL_ID",
                image_url=image_url,
            )

        else:
            pred = "__UNKNOWN__"

    except Exception as e:
        print(f"[错误] 调用模型 {model_name} 时出错: {e}")
        pred = "__UNKNOWN__"

    latency = time.time() - start
    return pred, latency
