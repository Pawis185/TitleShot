# hf_vlm_client.py
# 使用 Hugging Face InferenceClient 调用多模态模型（图片 + 文本）

import os
from typing import List, Optional
from huggingface_hub import InferenceClient

from config_hf import CATEGORIES, CLASSIFICATION_INSTRUCTION


def build_hf_client() -> InferenceClient:
    """
    构造一个 InferenceClient。
    - 默认会自动从本地环境读取 HF_TOKEN（也可以手动传 api_key）
    - 不指定 provider，交给 huggingface_hub 自动路由
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "未找到环境变量 HF_TOKEN，请先在系统中设置你的 Hugging Face 访问令牌。"
        )
    client = InferenceClient(api_key=token)
    return client


def _normalize_category(raw_text: str, categories: List[str]) -> Optional[str]:
    """
    把模型输出的文本，尽量映射到标准类别标签上。
    例如模型可能输出:
        "The category is Electronics."
        "类别：Home and Kitchen"
    我们需要映射回:
        "Electronics"
        "Home_and_Kitchen"
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    # 先处理一些常见前缀/符号
    for prefix in ["类别：", "类别:", "类：", "类:", "Category:", "category:", "类别为", "属于"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # 去掉句号之类
    text = text.replace("。", "").replace(".", "").strip()

    # 优先直接精确匹配
    if text in categories:
        return text

    lower_text = text.lower()

    # 尝试通过各种变体进行模糊匹配
    def gen_variants(cat: str):
        """
        为每个类别生成一些可能的变体，便于匹配：
        - 原始标签: "Home_and_Kitchen"
        - 小写: "home_and_kitchen"
        - 空格形式: "Home and Kitchen"
        - 小写空格形式: "home and kitchen"
        - 带 & 的形式: "Home & Kitchen"
        """
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

    for cat in categories:
        variants = gen_variants(cat)
        # 如果某个变体完整等于输出
        if text in variants or lower_text in variants:
            return cat

    # 再稍微宽松一点：如果输出文本中包含某个变体
    for cat in categories:
        variants = gen_variants(cat)
        for v in variants:
            if v and v.lower() in lower_text:
                return cat

    # 实在没法匹配，就返回 None，外面算作错误
    return None


def classify_image_with_model(
    client: InferenceClient,
    model_repo: str,
    image_url: str,
    categories: List[str] = CATEGORIES,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> Optional[str]:
    """
    调用某个多模态模型对图片进行分类。

    参数：
        client: 已经构造好的 InferenceClient
        model_repo: Hugging Face 上的模型 repo 名，例如 "Qwen/Qwen2-VL-7B-Instruct"
        image_url: 图片的 URL（dataset.json 里就是公网 URL）
        categories: 类别列表
    返回：
        预测的标准类别标签（匹配失败时返回 None）
    """

    # 构造分类任务提示词
    category_str = ", ".join(categories)
    user_text = CLASSIFICATION_INSTRUCTION.format(categories=category_str)

    # chat.completions 的多模态格式：
    # content 是一个 list，里面可以同时放 text 和 image_url 两种类型
    messages = [
        {
            "role": "system",
            "content": "你是一个电商商品图片分类助手。",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=model_repo,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 按照 Chat Completions 格式取出文本
    message = completion.choices[0].message
    # 有的实现是 message.content 是字符串，有的可能是 list，这里统一处理一下
    if isinstance(message.content, str):
        raw_text = message.content
    else:
        # 如果是 list[{"type":"text","text":...}, ...]
        text_parts = []
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        raw_text = "".join(text_parts)

    pred_label = _normalize_category(raw_text, categories)
    return pred_label
