# -*- coding: utf-8 -*-
"""
evaluate_perception.py

在 dataset.json 上评测 4 个感知层模型：
- Qwen3-VL-32B-Instruct
- InternVL2_5-8B
- MiniCPM-Llama3-V-2_5
- BLIP-2-OPT-2.7B-NLI

评测维度：
1. Overall Accuracy         整体准确率
2. Macro F1                 宏平均 F1
3. Per-class Accuracy       每类准确率
4. Valid Output Rate        输出合法类别比例（非 "__UNKNOWN__"）
5. Average Latency (s)      单样本平均推理时间
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import CATEGORIES, PERCEPTION_MODELS
from perception_models import classify_image


def load_dataset(dataset_path: str, max_per_class: int = None) -> List[Dict]:
    """
    读取 dataset.json，并展平成一个列表：
    [
        {"image_url": "...", "label": "Electronics"},
        ...
    ]

    参数：
        dataset_path  : json 文件路径
        max_per_class : 每个类别最多取多少条样本（None 表示全部使用）
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    per_class_counter = defaultdict(int)
    samples: List[Dict] = []

    for cate, items in raw.items():
        for item in items:
            if max_per_class is not None and per_class_counter[cate] >= max_per_class:
                continue
            samples.append(
                {
                    "image_url": item["image_url"],
                    "label": item["category"],
                }
            )
            per_class_counter[cate] += 1

    print(f"各类别采样数量: {dict(per_class_counter)}")
    print(f"总样本数: {len(samples)}")
    return samples


def compute_per_class_accuracy(
    y_true: List[str],
    y_pred: List[str],
    categories: List[str],
) -> Dict[str, float]:
    """
    计算每个类别的准确率：正确预测数 / 某类真实样本总数。
    """
    per_cls_correct = defaultdict(int)
    per_cls_total = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        per_cls_total[t] += 1
        if t == p:
            per_cls_correct[t] += 1

    per_cls_acc = {}
    for c in categories:
        if per_cls_total[c] == 0:
            per_cls_acc[c] = 0.0
        else:
            per_cls_acc[c] = per_cls_correct[c] / per_cls_total[c]
    return per_cls_acc


def evaluate_single_model(model_name: str, samples: List[Dict]) -> Dict:
    """
    对单个模型进行评测。
    """
    print(f"\n==== 开始评估模型: {model_name} ====")

    y_true: List[str] = []
    y_pred: List[str] = []
    latencies: List[float] = []

    valid_outputs = 0  # 非 "__UNKNOWN__" 的输出数量

    for sample in tqdm(samples, desc=f"Evaluating {model_name}", ncols=100):
        url = sample["image_url"]
        label = sample["label"]

        pred, latency = classify_image(model_name, url)
        y_true.append(label)
        y_pred.append(pred)
        latencies.append(latency)

        if pred in CATEGORIES:
            valid_outputs += 1

    # 1. 整体准确率
    acc = accuracy_score(y_true, y_pred)

    # 2. 宏平均 F1（只对合法类别计算）
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CATEGORIES,
        average="macro",
        zero_division=0,
    )

    # 3. 每类准确率
    per_cls_acc = compute_per_class_accuracy(y_true, y_pred, CATEGORIES)

    # 4. 有效输出比例
    valid_output_rate = valid_outputs / len(samples)

    # 5. 平均延时
    avg_latency = sum(latencies) / len(latencies)

    print(f"模型 {model_name} 评测结果：")
    print(f"  - Overall Accuracy      : {acc:.4f}")
    print(f"  - Macro F1              : {macro_f1:.4f}")
    print(f"  - Valid Output Rate     : {valid_output_rate:.4f}")
    print(f"  - Average Latency (sec) : {avg_latency:.3f}\n")

    print("  - Per-class Accuracy:")
    for c in CATEGORIES:
        print(f"    * {c:30s}: {per_cls_acc[c]:.4f}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "valid_output_rate": valid_output_rate,
        "avg_latency": avg_latency,
        "per_class_accuracy": per_cls_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset.json",
        help="数据集路径（默认当前目录下 dataset.json）",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="每个类别最多采样多少条（默认 None=全部）",
    )
    args = parser.parse_args()

    samples = load_dataset(args.dataset_path, args.max_per_class)

    all_results: List[Dict] = []
    best_model = None
    best_acc = -1.0

    for name in PERCEPTION_MODELS:
        try:
            r = evaluate_single_model(name, samples)
            all_results.append(r)
            if r["accuracy"] > best_acc:
                best_acc = r["accuracy"]
                best_model = r["model_name"]
        except Exception as e:
            print(f"⚠️ 模型 {name} 评估时发生异常：{e}，已跳过该模型。")

    print("\n==== 所有模型评估总结 ====")
    for r in all_results:
        print(
            f"- {r['model_name']}: "
            f"Acc={r['accuracy']:.4f}, "
            f"MacroF1={r['macro_f1']:.4f}, "
            f"ValidRate={r['valid_output_rate']:.4f}, "
            f"AvgLatency={r['avg_latency']:.3f}s"
        )

    if best_model is not None:
        print(
            f"\n>>> 表现最好的模型（按 Overall Accuracy 选）："
            f"{best_model}, accuracy={best_acc:.4f}"
        )


if __name__ == "__main__":
    main()
