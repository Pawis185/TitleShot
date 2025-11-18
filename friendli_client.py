# -*- coding: utf-8 -*-
"""
friendli_client.py

封装 Friendli Dedicated Endpoint 的调用（OpenAI 风格 chat.completions）：
- URL: https://api.friendli.ai/dedicated/v1/chat/completions
- model: 使用 endpoint 的 id（从环境变量 FRIENDLI_INTERNVL_MODEL_ID / FRIENDLI_MINICPM_MODEL_ID 中读）
"""

import os
import requests
from typing import List, Dict, Any


FRIENDLI_API_URL = "https://api.friendli.ai/dedicated/v1/chat/completions"


def call_friendli_chat_completion(
    model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 32,
) -> str:
    """
    调用 Friendli Dedicated Chat Completions 接口。

    参数：
        model_id    : Friendli endpoint 的 id，比如 "depx7m7tdfyych8"
        messages    : OpenAI 风格的 messages 列表
        temperature : 采样温度
        max_tokens  : 最大生成 token 数

    返回：
        模型生成的文本（第一条 message 的 content）
    """
    token = os.getenv("FRIENDLI_TOKEN")
    if not token:
        raise RuntimeError("未设置环境变量 FRIENDLI_TOKEN，无法访问 Friendli API")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(FRIENDLI_API_URL, headers=headers, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("[Friendli 错误响应]", resp.status_code, resp.text[:500])
        raise e

    data = resp.json()
    # OpenAI 风格返回：choices[0].message.content
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    # 兼容 content 是 [{type: "text", text: "..."}] 的情况
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "".join(texts)
    return str(content)
