"""
电商产品图像感知层模型评估系统 (本地模拟版)
作者: Shikang WANG

说明: 此版本使用模拟数据演示评估流程，实际部署时替换为真实API调用
"""

import json
import random
import time
from typing import Dict, List, Any
import numpy as np


# ==================== 模拟模型响应 ====================

def simulate_model_response(model_key: str, category: str, image_url: str) -> Dict[str, Any]:
    """
    模拟模型的API响应（用于演示）
    实际使用时应替换为真实的API调用

    Args:
        model_key: 模型标识符
        category: 真实类别
        image_url: 图片URL

    Returns:
        模拟的属性提取结果
    """
    # 不同模型的性能特征（模拟）
    model_performance = {
        "qwen3_vl_32b": {
            "category_accuracy": 0.92,
            "completeness": 0.95,
            "format_validity": 0.98,
            "latency_mean": 1.8,
            "latency_std": 0.3
        },
        "internvl2_5_8b": {
            "category_accuracy": 0.88,
            "completeness": 0.90,
            "format_validity": 0.95,
            "latency_mean": 1.2,
            "latency_std": 0.2
        },
        "minicpm_llama3_v_2_5": {
            "category_accuracy": 0.85,
            "completeness": 0.88,
            "format_validity": 0.92,
            "latency_mean": 1.0,
            "latency_std": 0.15
        },
        "blip2_opt_2_7b_nli": {
            "category_accuracy": 0.78,
            "completeness": 0.75,
            "format_validity": 0.85,
            "latency_mean": 0.8,
            "latency_std": 0.1
        }
    }

    perf = model_performance.get(model_key, model_performance["blip2_opt_2_7b_nli"])

    # 模拟延迟
    latency = max(0.1, random.gauss(perf["latency_mean"], perf["latency_std"]))
    time.sleep(latency)

    # 根据准确率决定是否预测正确的类别
    if random.random() < perf["category_accuracy"]:
        predicted_category = category
    else:
        # 预测错误的类别
        all_categories = ["Electronics", "Home & Kitchen", "Books", "Clothing & Jewelry",
                          "Sports & Outdoors", "Movies & TV", "Automotive", "Tools & Improvement"]
        wrong_categories = [c for c in all_categories if c != category]
        predicted_category = random.choice(wrong_categories)

    # 根据完整性决定包含哪些字段
    base_attributes = {
        "category": predicted_category,
        "main_product": f"{predicted_category} Product",
        "color": random.choice(["Black", "White", "Silver", "Blue", "Red"]),
        "material": random.choice(["Plastic", "Metal", "Wood", "Glass", "Fabric"]),
        "key_features": [
            "High quality construction",
            "Durable and long-lasting",
            "Easy to use"
        ],
        "selling_points": [
            "Premium quality materials",
            "Competitive pricing",
            "Fast shipping available"
        ],
        "target_audience": "General consumers",
        "usage_scenario": ["Daily use", "Professional use"]
    }

    # 根据完整性随机移除一些字段
    if random.random() > perf["completeness"]:
        fields_to_remove = random.sample(list(base_attributes.keys()),
                                         k=random.randint(1, 3))
        for field in fields_to_remove:
            base_attributes.pop(field, None)

    # 根据格式有效性决定是否返回格式正确的JSON
    if random.random() < perf["format_validity"]:
        return base_attributes
    else:
        # 返回格式有问题的数据
        corrupted = base_attributes.copy()
        if "key_features" in corrupted:
            corrupted["key_features"] = ", ".join(corrupted["key_features"])  # 应该是list但返回了string
        return corrupted


# ==================== 评估指标 ====================

def evaluate_category_accuracy(predicted_category: str, ground_truth_category: str) -> float:
    """评估类别识别准确率"""
    if not predicted_category:
        return 0.0

    pred = predicted_category.lower().strip()
    gt = ground_truth_category.lower().strip()

    if pred == gt:
        return 1.0
    elif pred in gt or gt in pred:
        return 0.5
    return 0.0


def evaluate_output_completeness(attributes: Dict[str, Any]) -> float:
    """评估输出完整性"""
    required_fields = [
        "category", "main_product", "color", "material",
        "key_features", "selling_points", "target_audience", "usage_scenario"
    ]

    if not attributes:
        return 0.0

    present_fields = sum(1 for field in required_fields
                         if field in attributes and attributes[field])
    return present_fields / len(required_fields)


def evaluate_selling_points_quality(selling_points: List[str]) -> float:
    """评估卖点提取质量"""
    if not selling_points or not isinstance(selling_points, list):
        return 0.0

    # 数量评分
    count_score = min(len(selling_points) / 5, 1.0) * 0.3

    # 长度评分
    lengths = [len(sp) for sp in selling_points if isinstance(sp, str)]
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        if 10 <= avg_length <= 50:
            length_score = 0.3
        elif 5 <= avg_length < 10 or 50 < avg_length <= 80:
            length_score = 0.15
        else:
            length_score = 0.0
    else:
        length_score = 0.0

    # 多样性评分
    if len(selling_points) > 1:
        unique_words = set()
        for sp in selling_points:
            if isinstance(sp, str):
                unique_words.update(sp.lower().split())
        diversity_score = min(len(unique_words) / (len(selling_points) * 3), 1.0) * 0.4
    else:
        diversity_score = 0.2

    return count_score + length_score + diversity_score


def evaluate_format_validity(attributes: Dict[str, Any]) -> float:
    """评估输出格式规范性"""
    if not isinstance(attributes, dict):
        return 0.0

    score = 0.0
    checks = [
        ("category", str),
        ("main_product", str),
        ("color", str),
        ("material", str),
        ("key_features", list),
        ("selling_points", list),
        ("target_audience", str),
        ("usage_scenario", list)
    ]

    for field, expected_type in checks:
        if isinstance(attributes.get(field), expected_type):
            score += 1

    return score / len(checks)


# ==================== 评估流程 ====================

def evaluate_single_sample(model_key: str, sample: Dict[str, Any],
                           sample_id: int) -> Dict[str, Any]:
    """评估单个样本"""
    print(f"  样本 {sample_id}: {sample['category']}")

    # 调用模型（模拟）
    start_time = time.time()
    attributes = simulate_model_response(model_key, sample["category"], sample["image_url"])
    latency = time.time() - start_time

    # 计算评估指标
    result = {
        "sample_id": sample_id,
        "success": True,
        "ground_truth_category": sample["category"],
        "predicted_attributes": attributes,
        "latency": latency,

        "category_accuracy": evaluate_category_accuracy(
            attributes.get("category", ""),
            sample["category"]
        ),
        "output_completeness": evaluate_output_completeness(attributes),
        "selling_points_quality": evaluate_selling_points_quality(
            attributes.get("selling_points", [])
        ),
        "format_validity": evaluate_format_validity(attributes)
    }

    print(f"    ✓ 类别: {attributes.get('category', 'N/A')} | "
          f"准确率: {result['category_accuracy']:.2f} | "
          f"完整性: {result['output_completeness']:.2f} | "
          f"延迟: {result['latency']:.2f}s")

    return result


def evaluate_model(model_key: str, model_name: str, test_samples: List[Dict[str, Any]],
                   max_samples: int = 20) -> Dict[str, Any]:
    """评估单个模型"""
    print(f"\n{'=' * 70}")
    print(f"正在评估模型: {model_name}")
    print(f"{'=' * 70}")

    samples_to_test = test_samples[:max_samples]
    results = []

    for i, sample in enumerate(samples_to_test, 1):
        result = evaluate_single_sample(model_key, sample, i)
        results.append(result)

        if i < len(samples_to_test):
            time.sleep(0.1)  # 短暂延迟

    # 汇总统计
    successful_results = [r for r in results if r["success"]]

    metrics = {
        "avg_category_accuracy": np.mean([r["category_accuracy"] for r in successful_results]),
        "avg_output_completeness": np.mean([r["output_completeness"] for r in successful_results]),
        "avg_selling_points_quality": np.mean([r["selling_points_quality"] for r in successful_results]),
        "avg_format_validity": np.mean([r["format_validity"] for r in successful_results]),
        "avg_latency": np.mean([r["latency"] for r in successful_results]),
        "success_rate": len(successful_results) / len(results)
    }

    # 综合得分
    metrics["overall_score"] = (
            metrics["avg_category_accuracy"] * 0.30 +  # 类别准确率 30%
            metrics["avg_output_completeness"] * 0.25 +  # 输出完整性 25%
            metrics["avg_selling_points_quality"] * 0.25 +  # 卖点质量 25%
            metrics["avg_format_validity"] * 0.20  # 格式规范性 20%
    )

    print(f"\n{'=' * 70}")
    print(f"模型 {model_name} 评估完成")
    print(f"{'=' * 70}")
    print(f"  成功率:       {metrics['success_rate']:.1%}")
    print(f"  类别准确率:   {metrics['avg_category_accuracy']:.1%}")
    print(f"  输出完整性:   {metrics['avg_output_completeness']:.1%}")
    print(f"  卖点质量:     {metrics['avg_selling_points_quality']:.1%}")
    print(f"  格式有效性:   {metrics['avg_format_validity']:.1%}")
    print(f"  平均延迟:     {metrics['avg_latency']:.3f}s")
    print(f"  综合得分:     {metrics['overall_score']:.1%}")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "total_samples": len(results),
        "successful_samples": len(successful_results),
        "metrics": metrics,
        "detailed_results": results
    }


def generate_evaluation_report(all_results: Dict[str, Any]):
    """生成评估报告"""
    print("\n" + "=" * 80)
    print("                     模型性能评估报告")
    print("=" * 80)

    # 按综合得分排序
    valid_results = {k: v for k, v in all_results.items() if "metrics" in v}
    sorted_models = sorted(valid_results.items(),
                           key=lambda x: x[1]["metrics"]["overall_score"],
                           reverse=True)

    # 打印对比表
    print(f"\n{'模型名称':<25} {'成功率':>8} {'类别准确':>10} {'完整性':>8} "
          f"{'卖点质量':>10} {'格式有效':>10} {'延迟(s)':>9} {'综合得分':>10}")
    print("-" * 105)

    for model_key, result in sorted_models:
        m = result["metrics"]
        print(f"{result['model_name']:<25} "
              f"{m['success_rate']:>7.1%} "
              f"{m['avg_category_accuracy']:>9.1%} "
              f"{m['avg_output_completeness']:>7.1%} "
              f"{m['avg_selling_points_quality']:>9.1%} "
              f"{m['avg_format_validity']:>9.1%} "
              f"{m['avg_latency']:>8.3f} "
              f"{m['overall_score']:>9.1%}")

    # 推荐模型
    if sorted_models:
        best_model_key, best_result = sorted_models[0]
        print(f"\n{'=' * 80}")
        print(f"推荐模型: {best_result['model_name']}")
        print(f"{'=' * 80}")

        m = best_result["metrics"]
        print("\n优势分析:")

        if m["avg_category_accuracy"] >= 0.90:
            print(f"  ✓ 类别识别准确率优秀 ({m['avg_category_accuracy']:.1%})")
        elif m["avg_category_accuracy"] >= 0.80:
            print(f"  ✓ 类别识别准确率良好 ({m['avg_category_accuracy']:.1%})")

        if m["avg_output_completeness"] >= 0.90:
            print(f"  ✓ 输出完整性优秀 ({m['avg_output_completeness']:.1%})")
        elif m["avg_output_completeness"] >= 0.80:
            print(f"  ✓ 输出完整性良好 ({m['avg_output_completeness']:.1%})")

        if m["avg_selling_points_quality"] >= 0.70:
            print(f"  ✓ 卖点提取质量优秀 ({m['avg_selling_points_quality']:.1%})")

        if m["avg_latency"] <= 2.0:
            print(f"  ✓ 响应速度快 ({m['avg_latency']:.3f}s)")

        print("\n改进建议:")

        if m["avg_category_accuracy"] < 0.85:
            print("  • 类别识别需要改进 - 建议:")
            print("    - 使用更详细的类别提示词")
            print("    - 提供类别示例")
            print("    - 考虑微调模型")

        if m["avg_output_completeness"] < 0.85:
            print("  • 输出完整性需要提升 - 建议:")
            print("    - 在提示词中明确列出所有必需字段")
            print("    - 使用结构化输出约束")
            print("    - 添加输出验证和补全机制")

        if m["avg_selling_points_quality"] < 0.65:
            print("  • 卖点提取质量需要改进 - 建议:")
            print("    - 提供卖点提取示例")
            print("    - 训练专门的卖点提取模型")
            print("    - 结合产品描述进行卖点增强")

        if m["avg_format_validity"] < 0.90:
            print("  • 输出格式规范性需要加强 - 建议:")
            print("    - 使用JSON Schema验证")
            print("    - 添加格式后处理步骤")
            print("    - 考虑使用结构化输出API")

    print(f"\n{'=' * 80}")
    print("下一步工作:")
    print("=" * 80)
    print("\n1. 针对最佳模型的不足之处，设计轻量级优化模型")
    print("   - 目标: 提升属性提取的准确性和一致性")
    print("   - 方法: 训练专门的属性分类和卖点提取模块")
    print("\n2. 建立两阶段处理流程:")
    print("   - 阶段1: 轻量级模型进行初步属性提取和prompt优化")
    print("   - 阶段2: 最佳VLM模型基于优化后的prompt生成最终结果")
    print("\n3. 性能优化:")
    print("   - 实现批处理以提高吞吐量")
    print("   - 添加缓存机制减少重复调用")
    print("   - 设计fallback机制保证可用性")

    print(f"\n{'=' * 80}\n")

    # 保存详细报告
    report_data = {
        "summary": {
            "best_model": best_result["model_name"] if sorted_models else None,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples_evaluated": best_result["total_samples"] if sorted_models else 0
        },
        "models": all_results
    }

    with open("/home/claude/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("详细评估报告已保存至: /home/claude/evaluation_report.json")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("           电商产品图像感知层模型评估系统 (模拟演示版)")
    print("=" * 80)
    print("\n说明: 此版本使用模拟数据演示评估流程")
    print("     实际使用时需要替换为真实API调用\n")

    # 加载数据集
    print("[1] 加载数据集...")
    with open("/mnt/user-data/uploads/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 准备测试样本
    test_samples = []
    samples_per_category = 3

    for category, items in dataset.items():
        sampled_items = items[:samples_per_category]
        test_samples.extend(sampled_items)

    print(f"准备了 {len(test_samples)} 个测试样本\n")

    # 模型配置
    models_to_evaluate = {
        "qwen3_vl_32b": "Qwen3-VL-32B-Instruct",
        "internvl2_5_8b": "InternVL2_5-8B",
        "minicpm_llama3_v_2_5": "MiniCPM-Llama3-V-2_5",
        "blip2_opt_2_7b_nli": "BLIP2-OPT-2.7B-NLI"
    }

    # 评估所有模型
    print("[2] 开始评估模型...")
    all_results = {}

    for model_key, model_name in models_to_evaluate.items():
        result = evaluate_model(model_key, model_name, test_samples, max_samples=12)
        all_results[model_key] = result
        time.sleep(0.5)

    # 生成报告
    print("\n[3] 生成评估报告...")
    generate_evaluation_report(all_results)

    print("评估完成!")


if __name__ == "__main__":
    main()