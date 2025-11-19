# evaluate_hf_models.py
# 在 dataset.json 上评估多个 Hugging Face 多模态模型

import argparse
from collections import defaultdict
from typing import List, Dict

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from config_hf import CATEGORIES, CANDIDATE_MODELS
from dataset_loader import load_dataset
from hf_vlm_client import build_hf_client, classify_image_with_model
from huggingface_hub.errors import BadRequestError, HfHubHTTPError, InferenceTimeoutError



def evaluate_single_model(
    model_name: str,
    model_repo: str,
    samples: List[Dict],
) -> Dict:
    """
    在给定样本上评估某一个模型。

    返回结果字典：
    {
        "model_name": ...,
        "model_repo": ...,
        "accuracy": float,
        "y_true": [...],
        "y_pred": [...]
    }
    """
    client = build_hf_client()

    y_true: List[str] = []
    y_pred: List[str] = []

    print(f"\n==== 开始评估模型: {model_name} ({model_repo}) ====")

    for sample in tqdm(samples, desc=f"Evaluating {model_name}", ncols=100):
        image_url = sample["image_url"]
        label = sample["label"]

        pred = classify_image_with_model(
            client=client,
            model_repo=model_repo,
            image_url=image_url,
            categories=CATEGORIES,
        )

        y_true.append(label)
        # 匹配失败就用特殊标签，后面会当作错误
        y_pred.append(pred if pred is not None else "__UNKNOWN__")

    # 把 "__UNKNOWN__" 视为错误类别
    cleaned_y_pred = [
        (p if p in CATEGORIES else "WRONG") for p in y_pred
    ]

    acc = accuracy_score(y_true, cleaned_y_pred)

    print(f"\n模型 {model_name} ({model_repo}) 的总体准确率: {acc:.4f}")
    print("\n分类报告（把错误预测统一归为 WRONG 类）：")
    print(
        classification_report(
            y_true,
            cleaned_y_pred,
            labels=CATEGORIES + ["WRONG"],
            zero_division=0,
        )
    )

    return {
        "model_name": model_name,
        "model_repo": model_repo,
        "accuracy": acc,
        "y_true": y_true,
        "y_pred": cleaned_y_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset.json",
        help="dataset.json 的路径",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="每个类别最多抽取多少样本（用于快速测试，None 表示用满全部样本）",
    )
    args = parser.parse_args()

    # 1. 加载数据集
    all_samples = load_dataset(args.dataset_path)

    # 2. 按类别分组（方便做下采样）
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for s in all_samples:
        by_label[s["label"]].append(s)

    eval_samples: List[Dict] = []
    for label, items in by_label.items():
        if args.max_samples_per_class is not None:
            eval_samples.extend(items[: args.max_samples_per_class])
        else:
            eval_samples.extend(items)

    print(
        f"总样本数: {len(eval_samples)} "
        f"(每类最多 {args.max_samples_per_class or '全部'} 条)"
    )

    # 3. 依次评估每个候选模型
    results: List[Dict] = []
    best_model = None
    best_acc = -1.0

    for name, repo in CANDIDATE_MODELS.items():
        try:
            result = evaluate_single_model(name, repo, eval_samples)

            # ❶ 明确的 BadRequest（比如 provider 不支持 / 健康检查失败）
        except BadRequestError as e:
            print(
                f"⚠️ 模型 {name} ({repo}) 调用失败：BadRequestError\n"
                f"   说明：当前 Hugging Face Inference Provider（例如 hyperbolic）"
                f" 不支持本次请求格式或健康检查不通过。\n"
                f"   详细错误：{e}\n"
                f"   → 已跳过该模型，继续评估下一个模型。\n"
            )
            continue

            # ❷ 其它 HTTP 相关错误（包括 5xx / router 问题等）
        except (HfHubHTTPError, InferenceTimeoutError) as e:
            print(
                f"⚠️ 模型 {name} ({repo}) 在 HF Inference 上调用失败：{type(e).__name__}\n"
                f"   详细错误：{e}\n"
                f"   → 已跳过该模型，继续评估下一个模型。\n"
            )
            continue

            # ❸ 兜底：任何其它未知异常
        except Exception as e:
            print(
                f"⚠️ 模型 {name} ({repo}) 评估时发生未知错误：{type(e).__name__}: {e}\n"
                f"   → 已跳过该模型，继续评估下一个模型。\n"
            )
            continue

        results.append(result)
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_model = result

    # 4. 汇总结果
    print("\n==== 所有模型评估完成 ====")
    for r in results:
        print(
            f"模型 {r['model_name']} ({r['model_repo']}): "
            f"accuracy = {r['accuracy']:.4f}"
        )

    if best_model is not None:
        print(
            f"\n>>> 表现最好的模型: {best_model['model_name']} "
            f"({best_model['model_repo']}), accuracy = {best_model['accuracy']:.4f}"
        )
        print("后续可以针对这个模型的错误样本进行分析，作为上层轻量模型训练的数据。")


if __name__ == "__main__":
    main()
