# -*- coding: utf-8 -*-
"""
blip2_client.py

BLIP-2 HF Inference Endpoint 调用封装。
同样使用 HF 默认的文本生成接口（不走 /v1/chat/completions）。
多模态也通过 prompt 中的图片 URL 实现。
"""

import os
import requests
from config import HF_BLIP2_ENDPOINT


def call_blip2_endpoint(
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> str:
    """
    调用 BLIP-2 HF Endpoint（文本接口）。

    参数：
        prompt         : 输入文本（包含图片 URL）
        max_new_tokens : 最多生成 token 数
        temperature    : 采样温度
    返回：
        模型生成的完整文本
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("未设置环境变量 HF_TOKEN，无法访问 Hugging Face Endpoint")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }

    resp = requests.post(HF_BLIP2_ENDPOINT, headers=headers, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("[BLIP-2 Endpoint 错误响应]", resp.status_code, resp.text[:500])
        raise e

    data = resp.json()

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return str(first)
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)
