"""
电商产品图像感知层模型评估系统 (完整版)
作者: Shikang WANG
功能: 评估 Qwen3-VL-32B, InternVL2.5-8B, MiniCPM-Llama3-V-2.5
"""

import json
import time
import base64
import requests
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from io import BytesIO
from PIL import Image
from pathlib import Path
import re
from collections import Counter
from dataclasses import dataclass, asdict

# ==================== 配置部分 ====================

# API 配置
API_CONFIG = {
    "qwen3_vl_32b": {
        "name": "Qwen3-VL-32B-Instruct",
        "endpoint": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "token": "sk-dbb9cb5041f641719b0daf6d4c0f67ed",
        "model_id": "qwen-vl-max",
        "max_tokens": 1024,
        "temperature": 0.1
    },
    "internvl2_5_8b": {
        "name": "InternVL2.5-8B",
        "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
        "token": "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b",
        "model_id": "depx7m7tdfyych8",
        "max_tokens": 1024,
        "temperature": 0.1
    },
    "minicpm_llama3_v_2_5": {
        "name": "MiniCPM-Llama3-V-2.5",
        "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
        "token": "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b",
        "model_id": "dep38olypw05c2m",
        "max_tokens": 1024,
        "temperature": 0.1
    }
}

# 标准化的属性提取提示词
ATTRIBUTE_EXTRACTION_PROMPT = """You are an expert e-commerce product analyst. Analyze this product image and extract key attributes.

Respond ONLY with valid JSON in this EXACT format (no additional text):
{
  "category": "primary product category (e.g., Electronics, Clothing, Home & Kitchen)",
  "main_product": "specific product name",
  "color": "primary color(s)",
  "material": "main material",
  "style": "design style (e.g., modern, casual, professional)",
  "key_features": ["feature1", "feature2", "feature3"],
  "selling_points": ["unique selling point 1", "unique selling point 2"],
  "target_audience": "target customer segment",
  "usage_scenario": ["scenario1", "scenario2"]
}

Requirements:
- selling_points must be 2-3 compelling marketing points (10-50 words each)
- All fields must be present
- Use clear, marketing-friendly language"""


# ==================== 数据类 ====================

@dataclass
class EvaluationMetrics:
    """单个样本的评估指标"""
    sample_id: int
    ground_truth_category: str
    predicted_category: str

    # 核心指标
    category_accuracy: float  # 0-1
    attribute_completeness: float  # 0-1
    selling_points_quality: float  # 0-1
    format_validity: float  # 0-1

    # 辅助指标
    latency: float  # seconds
    success: bool
    error_message: str = ""

    # 原始输出
    raw_output: str = ""
    parsed_attributes: Dict[str, Any] = None

    def overall_score(self) -> float:
        """计算综合得分 (加权平均)"""
        if not self.success:
            return 0.0
        return (
                self.category_accuracy * 0.30 +
                self.attribute_completeness * 0.25 +
                self.selling_points_quality * 0.25 +
                self.format_validity * 0.20
        )


@dataclass
class ModelPerformance:
    """模型整体性能"""
    model_name: str
    total_samples: int
    successful_samples: int

    # 平均指标
    avg_category_accuracy: float
    avg_attribute_completeness: float
    avg_selling_points_quality: float
    avg_format_validity: float
    avg_latency: float

    # 综合
    overall_score: float
    success_rate: float

    # 详细结果
    detailed_results: List[EvaluationMetrics] = None


# ==================== 工具函数 ====================

def download_and_encode_image(image_url: str, max_size: int = 1024) -> Tuple[str, bool]:
    """下载并编码图片为 base64"""
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        # 加载和预处理图片
        img = Image.open(BytesIO(response.content))

        # 调整大小
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 转换为 RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 编码为 JPEG base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_base64, True

    except Exception as e:
        print(f"      ⚠️  图片下载失败: {e}")
        return "", False


def call_model_api(model_key: str, image_base64: str, prompt: str) -> Tuple[str, float, str]:
    """调用模型 API (统一接口)"""
    config = API_CONFIG[model_key]
    start_time = time.time()

    headers = {
        "Authorization": f"Bearer {config['token']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config["model_id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"]
    }

    try:
        response = requests.post(
            config["endpoint"],
            headers=headers,
            json=payload,
            timeout=30
        )

        latency = time.time() - start_time

        if response.status_code != 200:
            return "", latency, f"API Error {response.status_code}: {response.text}"

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return content, latency, ""

    except Exception as e:
        latency = time.time() - start_time
        return "", latency, str(e)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """从文本中提取 JSON (鲁棒性解析)"""
    if not text:
        return {}

    # 尝试 1: 直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试 2: 移除 Markdown 代码块
    text_cleaned = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text_cleaned = re.sub(r'```\s*$', '', text_cleaned)
    try:
        return json.loads(text_cleaned.strip())
    except json.JSONDecodeError:
        pass

    # 尝试 3: 提取花括号内容
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 尝试 4: 手动解析 (启发式)
    result = {}
    patterns = {
        "category": r'["\']?category["\']?\s*:\s*["\']([^"\']+)["\']',
        "main_product": r'["\']?main_product["\']?\s*:\s*["\']([^"\']+)["\']',
        "color": r'["\']?color["\']?\s*:\s*["\']([^"\']+)["\']',
        "material": r'["\']?material["\']?\s*:\s*["\']([^"\']+)["\']',
        "style": r'["\']?style["\']?\s*:\s*["\']([^"\']+)["\']',
        "target_audience": r'["\']?target_audience["\']?\s*:\s*["\']([^"\']+)["\']',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # 提取数组字段
    for field in ["key_features", "selling_points", "usage_scenario"]:
        pattern = rf'["\']?{field}["\']?\s*:\s*\[(.*?)\]'
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            items_str = match.group(1)
            items = re.findall(r'["\']([^"\']+)["\']', items_str)
            result[field] = items

    return result


# ==================== 评估指标计算 ====================

def evaluate_category_accuracy(predicted: str, ground_truth: str) -> float:
    """评估类别识别准确率"""
    if not predicted:
        return 0.0

    pred = predicted.lower().strip().replace('_', ' ').replace('&', 'and')
    gt = ground_truth.lower().strip().replace('_', ' ').replace('&', 'and')

    # 完全匹配
    if pred == gt:
        return 1.0

    # 部分匹配
    if pred in gt or gt in pred:
        return 0.7

    # 词汇重叠
    pred_words = set(pred.split())
    gt_words = set(gt.split())
    overlap = len(pred_words & gt_words)
    if overlap > 0:
        return 0.4

    return 0.0


def evaluate_attribute_completeness(attributes: Dict[str, Any]) -> float:
    """评估属性提取完整性"""
    required_fields = [
        "category", "main_product", "color", "material", "style",
        "key_features", "selling_points", "target_audience", "usage_scenario"
    ]

    if not attributes:
        return 0.0

    score = 0.0
    for field in required_fields:
        if field in attributes and attributes[field]:
            value = attributes[field]

            # 列表字段需要有内容
            if isinstance(value, list):
                if len(value) > 0:
                    score += 1.0
            # 字符串字段需要有意义的内容
            elif isinstance(value, str):
                if len(value.strip()) >= 2:
                    score += 1.0

    return score / len(required_fields)


def evaluate_selling_points_quality(selling_points: List[str]) -> float:
    """评估卖点提取质量"""
    if not selling_points or not isinstance(selling_points, list):
        return 0.0

    # 1. 数量评分 (最佳 2-3 个)
    count = len(selling_points)
    if count == 0:
        count_score = 0.0
    elif 2 <= count <= 3:
        count_score = 1.0
    elif count == 1 or count == 4:
        count_score = 0.7
    else:
        count_score = 0.4

    # 2. 长度评分 (每个 10-50 词)
    lengths = []
    for sp in selling_points:
        if isinstance(sp, str):
            word_count = len(sp.split())
            lengths.append(word_count)

    if not lengths:
        length_score = 0.0
    else:
        avg_length = sum(lengths) / len(lengths)
        if 10 <= avg_length <= 50:
            length_score = 1.0
        elif 5 <= avg_length < 10 or 50 < avg_length <= 80:
            length_score = 0.6
        else:
            length_score = 0.2

    # 3. 多样性评分 (词汇丰富度)
    if len(selling_points) > 1:
        all_words = []
        for sp in selling_points:
            if isinstance(sp, str):
                words = sp.lower().split()
                all_words.extend(words)

        if all_words:
            unique_ratio = len(set(all_words)) / len(all_words)
            diversity_score = min(unique_ratio * 1.5, 1.0)
        else:
            diversity_score = 0.0
    else:
        diversity_score = 0.5

    # 加权平均
    return count_score * 0.3 + length_score * 0.4 + diversity_score * 0.3


def evaluate_format_validity(attributes: Dict[str, Any]) -> float:
    """评估输出格式规范性"""
    if not isinstance(attributes, dict):
        return 0.0

    expected_types = {
        "category": str,
        "main_product": str,
        "color": str,
        "material": str,
        "style": str,
        "key_features": list,
        "selling_points": list,
        "target_audience": str,
        "usage_scenario": list
    }

    score = 0.0
    for field, expected_type in expected_types.items():
        if field in attributes:
            value = attributes[field]
            if isinstance(value, expected_type):
                # 额外检查列表不为空
                if expected_type == list:
                    if len(value) > 0:
                        score += 1.0
                else:
                    score += 1.0

    return score / len(expected_types)


# ==================== 评估流程 ====================

def evaluate_single_sample(
        model_key: str,
        sample: Dict[str, Any],
        sample_id: int
) -> EvaluationMetrics:
    """评估单个样本"""
    ground_truth_category = sample["category"]
    image_url = sample["image_url"]

    # 下载图片
    image_base64, success = download_and_encode_image(image_url)
    if not success:
        return EvaluationMetrics(
            sample_id=sample_id,
            ground_truth_category=ground_truth_category,
            predicted_category="",
            category_accuracy=0.0,
            attribute_completeness=0.0,
            selling_points_quality=0.0,
            format_validity=0.0,
            latency=0.0,
            success=False,
            error_message="Failed to download image"
        )

    # 调用模型
    raw_output, latency, error = call_model_api(
        model_key,
        image_base64,
        ATTRIBUTE_EXTRACTION_PROMPT
    )

    if error:
        return EvaluationMetrics(
            sample_id=sample_id,
            ground_truth_category=ground_truth_category,
            predicted_category="",
            category_accuracy=0.0,
            attribute_completeness=0.0,
            selling_points_quality=0.0,
            format_validity=0.0,
            latency=latency,
            success=False,
            error_message=error,
            raw_output=raw_output
        )

    # 解析输出
    attributes = extract_json_from_text(raw_output)
    predicted_category = attributes.get("category", "")

    # 计算各项指标
    metrics = EvaluationMetrics(
        sample_id=sample_id,
        ground_truth_category=ground_truth_category,
        predicted_category=predicted_category,
        category_accuracy=evaluate_category_accuracy(predicted_category, ground_truth_category),
        attribute_completeness=evaluate_attribute_completeness(attributes),
        selling_points_quality=evaluate_selling_points_quality(attributes.get("selling_points", [])),
        format_validity=evaluate_format_validity(attributes),
        latency=latency,
        success=True,
        raw_output=raw_output,
        parsed_attributes=attributes
    )

    return metrics


def evaluate_model(
        model_key: str,
        test_samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None
) -> ModelPerformance:
    """评估单个模型"""
    model_name = API_CONFIG[model_key]["name"]

    print(f"\n{'=' * 80}")
    print(f"🔬 正在评估模型: {model_name}")
    print(f"{'=' * 80}")

    # 限制样本数量
    samples_to_test = test_samples[:max_samples] if max_samples else test_samples
    total_samples = len(samples_to_test)

    results: List[EvaluationMetrics] = []

    for idx, sample in enumerate(samples_to_test, 1):
        print(f"\n  📸 样本 {idx}/{total_samples}: {sample['category']}")
        print(f"      图片: {sample['image_url'][-50:]}")

        # 评估
        metrics = evaluate_single_sample(model_key, sample, idx)
        results.append(metrics)

        # 打印结果
        if metrics.success:
            print(f"      ✅ 成功")
            print(f"         类别: {metrics.predicted_category} (准确率: {metrics.category_accuracy:.2%})")
            print(f"         完整性: {metrics.attribute_completeness:.2%}")
            print(f"         卖点质量: {metrics.selling_points_quality:.2%}")
            print(f"         格式规范: {metrics.format_validity:.2%}")
            print(f"         延迟: {metrics.latency:.2f}s")
            print(f"         综合得分: {metrics.overall_score():.2%}")
        else:
            print(f"      ❌ 失败: {metrics.error_message}")

        # API 限流保护
        if idx < total_samples:
            time.sleep(1.5)

    # 计算整体性能
    successful_results = [r for r in results if r.success]

    if not successful_results:
        print("\n⚠️  警告: 所有样本都失败了!")
        return ModelPerformance(
            model_name=model_name,
            total_samples=total_samples,
            successful_samples=0,
            avg_category_accuracy=0.0,
            avg_attribute_completeness=0.0,
            avg_selling_points_quality=0.0,
            avg_format_validity=0.0,
            avg_latency=0.0,
            overall_score=0.0,
            success_rate=0.0,
            detailed_results=results
        )

    performance = ModelPerformance(
        model_name=model_name,
        total_samples=total_samples,
        successful_samples=len(successful_results),
        avg_category_accuracy=np.mean([r.category_accuracy for r in successful_results]),
        avg_attribute_completeness=np.mean([r.attribute_completeness for r in successful_results]),
        avg_selling_points_quality=np.mean([r.selling_points_quality for r in successful_results]),
        avg_format_validity=np.mean([r.format_validity for r in successful_results]),
        avg_latency=np.mean([r.latency for r in successful_results]),
        overall_score=np.mean([r.overall_score() for r in successful_results]),
        success_rate=len(successful_results) / total_samples,
        detailed_results=results
    )

    # 打印汇总
    print(f"\n{'=' * 80}")
    print(f"📊 模型 {model_name} 评估完成")
    print(f"{'=' * 80}")
    print(f"  成功率:         {performance.success_rate:.1%}")
    print(f"  类别准确率:     {performance.avg_category_accuracy:.1%}")
    print(f"  属性完整性:     {performance.avg_attribute_completeness:.1%}")
    print(f"  卖点质量:       {performance.avg_selling_points_quality:.1%}")
    print(f"  格式规范性:     {performance.avg_format_validity:.1%}")
    print(f"  平均延迟:       {performance.avg_latency:.2f}s")
    print(f"  综合得分:       {performance.overall_score:.1%}")
    print(f"{'=' * 80}")

    return performance


def generate_comparison_report(performances: Dict[str, ModelPerformance]) -> None:
    """生成对比报告"""
    print(f"\n{'=' * 80}")
    print("📈 模型性能对比报告")
    print(f"{'=' * 80}\n")

    # 按综合得分排序
    sorted_models = sorted(
        performances.items(),
        key=lambda x: x[1].overall_score,
        reverse=True
    )

    # 打印对比表
    print(f"{'模型名称':<30} {'成功率':>8} {'类别':>8} {'完整性':>8} {'卖点':>8} {'格式':>8} {'延迟':>8} {'综合':>8}")
    print("-" * 95)

    for model_key, perf in sorted_models:
        print(
            f"{perf.model_name:<30} "
            f"{perf.success_rate:>7.1%} "
            f"{perf.avg_category_accuracy:>7.1%} "
            f"{perf.avg_attribute_completeness:>7.1%} "
            f"{perf.avg_selling_points_quality:>7.1%} "
            f"{perf.avg_format_validity:>7.1%} "
            f"{perf.avg_latency:>7.2f}s "
            f"{perf.overall_score:>7.1%}"
        )

    # 推荐最佳模型
    if sorted_models:
        best_key, best_perf = sorted_models[0]
        print(f"\n{'=' * 80}")
        print(f"🏆 最佳模型: {best_perf.model_name}")
        print(f"{'=' * 80}")
        print(f"\n综合得分: {best_perf.overall_score:.1%}")
        print(f"\n优势:")

        if best_perf.avg_category_accuracy >= 0.85:
            print(f"  ✅ 类别识别优秀 ({best_perf.avg_category_accuracy:.1%})")
        if best_perf.avg_attribute_completeness >= 0.85:
            print(f"  ✅ 属性提取完整 ({best_perf.avg_attribute_completeness:.1%})")
        if best_perf.avg_selling_points_quality >= 0.70:
            print(f"  ✅ 卖点质量高 ({best_perf.avg_selling_points_quality:.1%})")
        if best_perf.avg_latency <= 3.0:
            print(f"  ✅ 响应速度快 ({best_perf.avg_latency:.2f}s)")

        print(f"\n改进方向:")
        if best_perf.avg_category_accuracy < 0.85:
            print(f"  🔧 类别识别需提升 (当前 {best_perf.avg_category_accuracy:.1%})")
        if best_perf.avg_attribute_completeness < 0.85:
            print(f"  🔧 属性完整性需提升 (当前 {best_perf.avg_attribute_completeness:.1%})")
        if best_perf.avg_selling_points_quality < 0.70:
            print(f"  🔧 卖点质量需提升 (当前 {best_perf.avg_selling_points_quality:.1%})")

        print(f"\n{'=' * 80}")
        print("💡 下一步优化方案:")
        print(f"{'=' * 80}")
        print("\n1. 针对最佳模型的不足，训练轻量级优化模型:")
        print("   - 如果类别识别弱 → 训练分类器微调")
        print("   - 如果卖点质量弱 → 训练卖点生成模型")
        print("   - 如果格式不规范 → 添加后处理规则")

        print("\n2. 两阶段优化流程:")
        print("   阶段1: 轻量级模型优化 Prompt")
        print("   阶段2: 最佳模型生成最终输出")

        print("\n3. 实施建议:")
        print("   - 使用小型 BERT/T5 模型进行属性分类")
        print("   - 设计模板库和规则引擎")
        print("   - 添加输出验证和自动修正")


def save_evaluation_results(
        performances: Dict[str, ModelPerformance],
        output_path: str = "evaluation_results.json"
) -> None:
    """保存评估结果到 JSON"""
    results = {}

    for model_key, perf in performances.items():
        results[model_key] = {
            "model_name": perf.model_name,
            "summary": {
                "total_samples": perf.total_samples,
                "successful_samples": perf.successful_samples,
                "success_rate": perf.success_rate,
                "avg_category_accuracy": perf.avg_category_accuracy,
                "avg_attribute_completeness": perf.avg_attribute_completeness,
                "avg_selling_points_quality": perf.avg_selling_points_quality,
                "avg_format_validity": perf.avg_format_validity,
                "avg_latency": perf.avg_latency,
                "overall_score": perf.overall_score
            },
            "detailed_results": [
                {
                    "sample_id": r.sample_id,
                    "success": r.success,
                    "ground_truth": r.ground_truth_category,
                    "predicted": r.predicted_category,
                    "metrics": {
                        "category_accuracy": r.category_accuracy,
                        "attribute_completeness": r.attribute_completeness,
                        "selling_points_quality": r.selling_points_quality,
                        "format_validity": r.format_validity,
                        "overall_score": r.overall_score()
                    },
                    "latency": r.latency,
                    "error": r.error_message
                }
                for r in perf.detailed_results
            ]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 详细结果已保存至: {output_path}")


# ==================== 主函数 ====================

def main():
    """主评估流程"""
    print("\n" + "=" * 80)
    print("🚀 TitleShot 感知层模型评估系统")
    print("=" * 80)
    print("\n作者: Shikang WANG")
    print("任务: 评估并选择最佳视觉-语言模型\n")

    # 1. 加载数据集
    print("[1/4] 加载数据集...")
    dataset_path = "dataset.json"

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 {dataset_path}")
        print("   请确保 dataset.json 在当前目录下")
        return

    # 准备测试样本 (每个类别取 10 张)
    test_samples = []
    samples_per_category = 10

    for category, items in dataset.items():
        selected = items[:samples_per_category]
        test_samples.extend(selected)

    total_samples = len(test_samples)
    print(f"✅ 已加载 {len(dataset)} 个类别, 共 {total_samples} 个测试样本\n")

    # 2. 评估所有模型
    print("[2/4] 开始评估模型...")
    print(f"{'=' * 80}\n")

    models_to_evaluate = [
        "qwen3_vl_32b",
        "internvl2_5_8b",
        "minicpm_llama3_v_2_5"
    ]

    performances = {}

    for model_key in models_to_evaluate:
        try:
            perf = evaluate_model(
                model_key,
                test_samples,
                max_samples=None  # 评估所有样本，如需快速测试可设为 10
            )
            performances[model_key] = perf

            # 模型之间的间隔
            time.sleep(2)

        except Exception as e:
            print(f"\n❌ 模型 {model_key} 评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 3. 生成对比报告
    print("\n[3/4] 生成对比报告...")
    if performances:
        generate_comparison_report(performances)
    else:
        print("❌ 没有成功评估的模型")
        return

    # 4. 保存结果
    print("\n[4/4] 保存评估结果...")
    save_evaluation_results(performances)

    print("\n" + "=" * 80)
    print("✅ 评估完成!")
    print("=" * 80)
    print("\n📝 后续步骤:")
    print("   1. 查看 evaluation_results.json 了解详细结果")
    print("   2. 根据最佳模型的不足设计优化方案")
    print("   3. 训练轻量级优化模型改进输出质量")
    print("   4. 实现两阶段处理流程")


if __name__ == "__main__":
    main()