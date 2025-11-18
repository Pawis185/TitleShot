# -*- coding: utf-8 -*-
"""
qwen3_client.py

Qwen/Qwen3-VL-32B-Instruct 的 Hugging Face Endpoint 调用封装。
注意：
  - 这里不使用 /v1/chat/completions，而是 HF 默认的文本生成接口：
      POST {HF_QWEN3_ENDPOINT}
      JSON: {"inputs": prompt, "parameters": {...}}
  - 多模态部分通过在 prompt 里写入图片 URL 来实现（软多模态）。
"""

import os
import requests
from config import HF_QWEN3_ENDPOINT


def call_qwen3_vl_endpoint(
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> str:
    """
    调用 Qwen3 HF Inference Endpoint（文本接口）。

    参数：
        prompt         : 输入给模型的文本
        max_new_tokens : 最多生成 token 数
        temperature    : 采样温度
    返回：
        模型生成的完整文本（字符串）
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

    resp = requests.post(HF_QWEN3_ENDPOINT, headers=headers, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("[Qwen3 Endpoint 错误响应]", resp.status_code, resp.text[:500])
        raise e

    data = resp.json()

    # 文本生成典型返回格式： [{"generated_text": "..."}]
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return str(first)
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)
