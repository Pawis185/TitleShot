# -*- coding: utf-8 -*-
"""
prompt_router.py

本地轻量级“上层模型”，职责：
- 输入：感知层输出的类别 + 场景标签（scenario）
- 输出：一个适合当前商品的 Prompt 文本（给下游大模型用）

技术实现：
- 使用 sklearn LogisticRegression 做多分类
- 特征：category（类别）、scenario（使用场景）
- 标签：prompt_style（不同风格的 Prompt 模板 ID）
"""

import argparse
import json
import joblib
from typing import List, Dict

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import CATEGORIES

MODEL_PATH = "prompt_router_model.joblib"
LABEL_ENCODER_PATH = "prompt_router_label_encoder.joblib"


# =========================
# Prompt 模板字典（可根据作业需要自行扩展）
# =========================

PROMPT_TEMPLATES: Dict[str, str] = {
    "short_promo": (
        "你是一个电商运营专家，请根据以下商品属性生成一个精简、突出卖点的商品标题和 3 条要点式卖点描述。"
        "风格要求：简短有力、适合放在 Amazon 标题栏，包含 1~2 个核心功能或卖点，以及 1 个促销性质的词语。"
        "\n\n商品类别：{category}\n使用场景：{scenario}\n"
    ),
    "scenario_lifestyle": (
        "你是一个擅长场景化文案的营销专家，请根据商品信息生成："
        "1 个有生活气息的标题 + 一段 80~120 字的场景化描述，再加 3 个相关话题标签（# 开头）。"
        "文案要体现具体使用情景（如通勤、居家、旅行），语言自然、让人产生购买冲动。"
        "\n\n商品类别：{category}\n使用场景：{scenario}\n"
    ),
    "storytelling": (
        "请为以下商品写一个带有故事感的文案，包括："
        "1 个标题 + 1 段 100~150 字的故事性描述。"
        "需要体现商品如何解决用户某个具体痛点，并通过情绪共鸣来提升购买意愿。"
        "\n\n商品类别：{category}\n使用场景：{scenario}\n"
    ),
    "tech_detail": (
        "请生成一个偏技术细节和规格说明的商品文案，包含："
        "1 个标题 + 5 条要点式说明（突出参数、性能、材质、兼容性等），"
        "适合对比同类产品时使用。"
        "\n\n商品类别：{category}\n使用场景：{scenario}\n"
    ),
}


# =========================
# 训练逻辑
# =========================

def train_router(train_file: str):
    """
    训练 Prompt Router 模型。

    训练数据格式（jsonl，每行一条）：
        {"category": "Electronics", "scenario": "promotion", "prompt_style": "short_promo"}
    """
    xs: List[Dict] = []
    ys: List[str] = []

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cat = obj["category"]
            scen = obj.get("scenario", "default")
            style = obj["prompt_style"]

            xs.append({"category": cat, "scenario": scen})
            ys.append(style)

    if not xs:
        raise RuntimeError("训练数据为空，请检查 train_file。")

    # LabelEncoder 把 prompt_style（字符串）编码成整数标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(ys)

    # Pipeline: DictVectorizer + LogisticRegression
    model = Pipeline(
        [
            ("vec", DictVectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto",
                ),
            ),
        ]
    )

    model.fit(xs, y_encoded)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"✅ 训练完成，模型已保存到：{MODEL_PATH}")
    print(f"✅ LabelEncoder 已保存到：{LABEL_ENCODER_PATH}")
    print(f"共训练样本数: {len(xs)}，prompt_style 类别数: {len(le.classes_)}")


# =========================
# 推理逻辑：生成 Prompt 文本
# =========================

def load_router():
    """加载训练好的 Prompt Router 模型和 LabelEncoder。"""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, le


def generate_prompt(category: str, scenario: str) -> str:
    """
    给定感知层输出的类别 + 场景标签，返回一个适合的 Prompt 文本。

    category : 图像感知层输出的类别（需在 CATEGORIES 中）
    scenario : 使用场景标签（自定义，如 promotion / daily_wear / gift 等）
    """
    if category not in CATEGORIES:
        raise ValueError(f"非法类别: {category}，请使用 config.CATEGORIES 中的值。")

    model, le = load_router()

    x = [{"category": category, "scenario": scenario}]
    y_pred_encoded = model.predict(x)[0]
    style = le.inverse_transform([y_pred_encoded])[0]  # 反编码到 prompt_style 字符串

    template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["short_promo"])
    prompt = template.format(category=category, scenario=scenario)
    return prompt


# =========================
# 命令行接口
# =========================

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # 子命令：train
    train_parser = subparsers.add_parser("train", help="训练 Prompt Router")
    train_parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="训练数据 jsonl 文件路径（每行一个 JSON）",
    )

    # 子命令：infer
    infer_parser = subparsers.add_parser("infer", help="根据类别+场景生成 Prompt")
    infer_parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="商品类别（需在 config.CATEGORIES 中）",
    )
    infer_parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        help="使用场景标签（自定义字符串，如 promotion / daily_wear 等）",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_router(args.train_file)
    elif args.command == "infer":
        prompt = generate_prompt(args.category, args.scenario)
        print("\n=== 生成的 Prompt 文本 ===\n")
        print(prompt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
