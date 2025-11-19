"""
电商产品图像感知层模型评估系统
作者: Shikang WANG
功能: 评估多个视觉-语言模型在产品属性提取任务上的性能
"""

import json
import os
import time
import base64
import requests
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
from io import BytesIO
from PIL import Image

# ==================== 配置部分 ====================

# API密钥配置
HF_TOKEN = "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"
FRIENDLI_TOKEN = "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b"
FRIENDLI_INTERNVL_MODEL_ID = "depx7m7tdfyych8"
FRIENDLI_MINICPM_MODEL_ID = "dep38olypw05c2m"

# 模型API端点配置
MODEL_CONFIGS = {
    "qwen3_vl_32b": {
        "name": "Qwen3-VL-32B-Instruct",
        "endpoint": "https://bfhfmuskpdnp4ajz.us-east-1.aws.endpoints.huggingface.cloud",
        "api_type": "huggingface",
        "token": HF_TOKEN
    },
    "internvl2_5_8b": {
        "name": "InternVL2_5-8B",
        "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
        "api_type": "friendli",
        "token": FRIENDLI_TOKEN,
        "model_id": FRIENDLI_INTERNVL_MODEL_ID
    },
    "minicpm_llama3_v_2_5": {
        "name": "MiniCPM-Llama3-V-2_5",
        "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
        "api_type": "friendli",
        "token": FRIENDLI_TOKEN,
        "model_id": FRIENDLI_MINICPM_MODEL_ID
    },
    "blip2_opt_2_7b_nli": {
        "name": "BLIP2-OPT-2.7B-NLI",
        "endpoint": "https://onnunaok1tz3tbo2.us-east-1.aws.endpoints.huggingface.cloud",
        "api_type": "huggingface",
        "token": HF_TOKEN
    }
}

# ==================== 提示词模板 ====================

# 产品属性提取提示词（结构化输出）
ATTRIBUTE_EXTRACTION_PROMPT = """请仔细分析这张电商产品图片，并提取以下结构化信息。

你需要以JSON格式输出结果，包含以下字段：
1. category: 产品类别（如Electronics, Clothing, Home & Kitchen等）
2. main_product: 主要产品名称
3. color: 主要颜色（可以是多个，用逗号分隔）
4. material: 材质（如果能识别）
5. key_features: 关键特征列表（3-5个要点）
6. selling_points: 营销卖点列表（3-5个）
7. target_audience: 目标受众
8. usage_scenario: 使用场景（1-2个）

请严格按照以下JSON格式输出，不要添加任何其他文字：
{
  "category": "产品类别",
  "main_product": "产品名称",
  "color": "颜色",
  "material": "材质",
  "key_features": ["特征1", "特征2", "特征3"],
  "selling_points": ["卖点1", "卖点2", "卖点3"],
  "target_audience": "目标人群",
  "usage_scenario": ["场景1", "场景2"]
}"""


# ==================== 工具函数 ====================

def download_and_encode_image(image_url: str, max_retries: int = 3) -> Tuple[str, bool]:
    """
    下载图片并转换为base64编码

    Args:
        image_url: 图片URL
        max_retries: 最大重试次数

    Returns:
        (base64_string, success): base64编码的图片和成功标志
    """
    for attempt in range(max_retries):
        try:
            # 下载图片
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # 转换为PIL Image以验证和可能的格式转换
            img = Image.open(BytesIO(response.content))

            # 如果图片太大，调整大小以节省token
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为RGB（如果是RGBA或其他模式）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 保存为JPEG并编码为base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return img_base64, True

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[警告] 下载图片失败 {image_url}: {str(e)}")
                return "", False
            time.sleep(1)  # 重试前等待

    return "", False


def call_huggingface_endpoint(endpoint: str, token: str, image_base64: str,
                              prompt: str, timeout: int = 30) -> Dict[str, Any]:
    """
    调用HuggingFace Inference Endpoint

    Args:
        endpoint: API端点URL
        token: 认证token
        image_base64: base64编码的图片
        prompt: 提示词
        timeout: 超时时间

    Returns:
        API响应的字典
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 对于视觉-语言模型，需要同时发送图片和文本
    payload = {
        "inputs": {
            "image": image_base64,
            "text": prompt
        },
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"HuggingFace API调用失败: {str(e)}")


def call_friendli_endpoint(endpoint: str, token: str, model_id: str,
                           image_base64: str, prompt: str, timeout: int = 30) -> Dict[str, Any]:
    """
    调用Friendli AI Dedicated Endpoint

    Args:
        endpoint: API端点URL
        token: 认证token
        model_id: 模型ID
        image_base64: base64编码的图片
        prompt: 提示词
        timeout: 超时时间

    Returns:
        API响应的字典
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Friendli API使用OpenAI兼容格式
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Friendli API调用失败: {str(e)}")


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    从模型响应中提取JSON内容

    Args:
        response_text: 模型的文本响应

    Returns:
        解析后的JSON字典
    """
    # 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 尝试提取代码块中的JSON
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取大括号包围的内容
    brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 如果都失败，返回空字典
    return {}


def call_model_api(model_key: str, image_base64: str, prompt: str) -> Tuple[Dict[str, Any], float, str]:
    """
    调用指定模型的API

    Args:
        model_key: 模型标识符
        image_base64: base64编码的图片
        prompt: 提示词

    Returns:
        (提取的属性字典, 响应时间, 错误信息)
    """
    config = MODEL_CONFIGS[model_key]
    start_time = time.time()

    try:
        if config["api_type"] == "huggingface":
            response = call_huggingface_endpoint(
                config["endpoint"],
                config["token"],
                image_base64,
                prompt
            )

            # 解析HuggingFace响应
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get("generated_text", "")
            elif isinstance(response, dict):
                response_text = response.get("generated_text", "")
            else:
                response_text = str(response)

        elif config["api_type"] == "friendli":
            response = call_friendli_endpoint(
                config["endpoint"],
                config["token"],
                config["model_id"],
                image_base64,
                prompt
            )

            # 解析Friendli响应（OpenAI格式）
            if "choices" in response and len(response["choices"]) > 0:
                response_text = response["choices"][0]["message"]["content"]
            else:
                response_text = str(response)
        else:
            return {}, 0, "未知的API类型"

        # 提取JSON
        extracted_attrs = extract_json_from_response(response_text)
        latency = time.time() - start_time

        return extracted_attrs, latency, ""

    except Exception as e:
        latency = time.time() - start_time
        return {}, latency, str(e)


# ==================== 评估指标 ====================

def evaluate_category_accuracy(predicted_category: str, ground_truth_category: str) -> float:
    """
    评估类别识别准确率

    Args:
        predicted_category: 预测的类别
        ground_truth_category: 真实类别

    Returns:
        准确率分数（0或1）
    """
    if not predicted_category:
        return 0.0

    # 标准化比较（忽略大小写和空格）
    pred = predicted_category.lower().strip()
    gt = ground_truth_category.lower().strip()

    # 完全匹配
    if pred == gt:
        return 1.0

    # 部分匹配（考虑类别名称可能的变体）
    if pred in gt or gt in pred:
        return 0.5

    return 0.0


def evaluate_output_completeness(attributes: Dict[str, Any]) -> float:
    """
    评估输出完整性（是否包含所有必需字段）

    Args:
        attributes: 提取的属性字典

    Returns:
        完整性分数（0-1）
    """
    required_fields = [
        "category", "main_product", "color", "material",
        "key_features", "selling_points", "target_audience", "usage_scenario"
    ]

    if not attributes:
        return 0.0

    present_fields = sum(1 for field in required_fields if field in attributes and attributes[field])
    return present_fields / len(required_fields)


def evaluate_selling_points_quality(selling_points: List[str]) -> float:
    """
    评估卖点提取质量

    Args:
        selling_points: 提取的卖点列表

    Returns:
        质量分数（0-1）
    """
    if not selling_points or not isinstance(selling_points, list):
        return 0.0

    # 基础分数：根据卖点数量
    count_score = min(len(selling_points) / 5, 1.0) * 0.3

    # 长度分数：卖点应该有适当的长度（不能太短也不能太长）
    lengths = [len(sp) for sp in selling_points if isinstance(sp, str)]
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        # 理想长度在10-50字符之间
        if 10 <= avg_length <= 50:
            length_score = 0.3
        elif 5 <= avg_length < 10 or 50 < avg_length <= 80:
            length_score = 0.15
        else:
            length_score = 0.0
    else:
        length_score = 0.0

    # 多样性分数：卖点之间应该有差异
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
    """
    评估输出格式规范性

    Args:
        attributes: 提取的属性字典

    Returns:
        格式有效性分数（0-1）
    """
    if not isinstance(attributes, dict):
        return 0.0

    score = 0.0
    total_checks = 8

    # 检查各字段类型
    if isinstance(attributes.get("category"), str):
        score += 1
    if isinstance(attributes.get("main_product"), str):
        score += 1
    if isinstance(attributes.get("color"), str):
        score += 1
    if isinstance(attributes.get("material"), str):
        score += 1
    if isinstance(attributes.get("key_features"), list):
        score += 1
    if isinstance(attributes.get("selling_points"), list):
        score += 1
    if isinstance(attributes.get("target_audience"), str):
        score += 1
    if isinstance(attributes.get("usage_scenario"), list):
        score += 1

    return score / total_checks


# ==================== 主评估流程 ====================

def evaluate_single_sample(model_key: str, sample: Dict[str, Any],
                           sample_id: int) -> Dict[str, Any]:
    """
    评估单个样本

    Args:
        model_key: 模型标识符
        sample: 样本数据（包含category和image_url）
        sample_id: 样本ID

    Returns:
        评估结果字典
    """
    print(f"  样本 {sample_id}: {sample['image_url'][:60]}...")

    # 下载并编码图片
    image_base64, success = download_and_encode_image(sample["image_url"])
    if not success:
        return {
            "sample_id": sample_id,
            "success": False,
            "error": "图片下载失败"
        }

    # 调用模型API
    attributes, latency, error = call_model_api(
        model_key,
        image_base64,
        ATTRIBUTE_EXTRACTION_PROMPT
    )

    if error:
        return {
            "sample_id": sample_id,
            "success": False,
            "error": error,
            "latency": latency
        }

    # 计算各项指标
    result = {
        "sample_id": sample_id,
        "success": True,
        "ground_truth_category": sample["category"],
        "predicted_attributes": attributes,
        "latency": latency,

        # 评估指标
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

    print(f"    ✓ 类别准确率: {result['category_accuracy']:.2f} | "
          f"完整性: {result['output_completeness']:.2f} | "
          f"延迟: {result['latency']:.2f}s")

    return result


def evaluate_model(model_key: str, test_samples: List[Dict[str, Any]],
                   max_samples: int = 20) -> Dict[str, Any]:
    """
    评估单个模型

    Args:
        model_key: 模型标识符
        test_samples: 测试样本列表
        max_samples: 最大测试样本数

    Returns:
        模型评估结果
    """
    model_name = MODEL_CONFIGS[model_key]["name"]
    print(f"\n{'=' * 60}")
    print(f"正在评估模型: {model_name}")
    print(f"{'=' * 60}")

    # 限制样本数量
    samples_to_test = test_samples[:max_samples]
    results = []

    # 逐个评估样本
    for i, sample in enumerate(samples_to_test, 1):
        result = evaluate_single_sample(model_key, sample, i)
        results.append(result)

        # 避免请求过快
        if i < len(samples_to_test):
            time.sleep(0.5)

    # 汇总统计
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print(f"\n[警告] 模型 {model_name} 没有成功的预测结果！")
        return {
            "model_key": model_key,
            "model_name": model_name,
            "total_samples": len(results),
            "successful_samples": 0,
            "metrics": {}
        }

    metrics = {
        "avg_category_accuracy": np.mean([r["category_accuracy"] for r in successful_results]),
        "avg_output_completeness": np.mean([r["output_completeness"] for r in successful_results]),
        "avg_selling_points_quality": np.mean([r["selling_points_quality"] for r in successful_results]),
        "avg_format_validity": np.mean([r["format_validity"] for r in successful_results]),
        "avg_latency": np.mean([r["latency"] for r in successful_results]),
        "success_rate": len(successful_results) / len(results)
    }

    # 计算综合得分
    metrics["overall_score"] = (
            metrics["avg_category_accuracy"] * 0.3 +
            metrics["avg_output_completeness"] * 0.25 +
            metrics["avg_selling_points_quality"] * 0.25 +
            metrics["avg_format_validity"] * 0.2
    )

    print(f"\n模型 {model_name} 评估完成:")
    print(f"  - 成功率: {metrics['success_rate']:.2%}")
    print(f"  - 类别准确率: {metrics['avg_category_accuracy']:.2%}")
    print(f"  - 输出完整性: {metrics['avg_output_completeness']:.2%}")
    print(f"  - 卖点质量: {metrics['avg_selling_points_quality']:.2%}")
    print(f"  - 格式有效性: {metrics['avg_format_validity']:.2%}")
    print(f"  - 平均延迟: {metrics['avg_latency']:.2f}s")
    print(f"  - 综合得分: {metrics['overall_score']:.2%}")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "total_samples": len(results),
        "successful_samples": len(successful_results),
        "metrics": metrics,
        "detailed_results": results
    }


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("电商产品图像感知层模型评估系统")
    print("=" * 60)

    # 加载数据集
    print("\n[1] 加载数据集...")
    with open("/mnt/user-data/uploads/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 准备测试样本（从每个类别中抽取样本）
    test_samples = []
    samples_per_category = 5  # 每个类别抽取5个样本

    for category, items in dataset.items():
        sampled_items = items[:samples_per_category]
        test_samples.extend(sampled_items)

    print(f"总共准备了 {len(test_samples)} 个测试样本")

    # 评估所有模型
    print("\n[2] 开始评估模型...")
    all_results = {}

    for model_key in MODEL_CONFIGS.keys():
        try:
            result = evaluate_model(model_key, test_samples, max_samples=10)
            all_results[model_key] = result
        except Exception as e:
            print(f"\n[错误] 评估模型 {model_key} 时出错: {str(e)}")
            all_results[model_key] = {
                "model_key": model_key,
                "model_name": MODEL_CONFIGS[model_key]["name"],
                "error": str(e)
            }

    # 生成评估报告
    print("\n[3] 生成评估报告...")
    generate_evaluation_report(all_results)

    print("\n评估完成！")


def generate_evaluation_report(all_results: Dict[str, Any]):
    """
    生成评估报告

    Args:
        all_results: 所有模型的评估结果
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("电商产品图像感知层模型评估报告")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 模型对比表
    report_lines.append("## 模型性能对比")
    report_lines.append("")
    report_lines.append(f"{'模型名称':<30} {'成功率':<10} {'类别准确率':<12} {'完整性':<10} "
                        f"{'卖点质量':<10} {'格式有效性':<12} {'平均延迟':<10} {'综合得分':<10}")
    report_lines.append("-" * 110)

    # 排序模型（按综合得分）
    valid_results = {k: v for k, v in all_results.items()
                     if "metrics" in v and v["metrics"]}
    sorted_models = sorted(valid_results.items(),
                           key=lambda x: x[1]["metrics"].get("overall_score", 0),
                           reverse=True)

    for model_key, result in sorted_models:
        metrics = result["metrics"]
        report_lines.append(
            f"{result['model_name']:<30} "
            f"{metrics['success_rate']:>8.1%}  "
            f"{metrics['avg_category_accuracy']:>10.1%}  "
            f"{metrics['avg_output_completeness']:>8.1%}  "
            f"{metrics['avg_selling_points_quality']:>8.1%}  "
            f"{metrics['avg_format_validity']:>10.1%}  "
            f"{metrics['avg_latency']:>8.2f}s  "
            f"{metrics['overall_score']:>8.1%}"
        )

    report_lines.append("")
    report_lines.append("## 推荐模型")
    report_lines.append("")

    if sorted_models:
        best_model_key, best_result = sorted_models[0]
        report_lines.append(f"根据综合评估，推荐使用: **{best_result['model_name']}**")
        report_lines.append("")
        report_lines.append("推荐理由:")

        metrics = best_result["metrics"]
        if metrics["avg_category_accuracy"] >= 0.8:
            report_lines.append(f"  ✓ 类别识别准确率高 ({metrics['avg_category_accuracy']:.1%})")
        if metrics["avg_output_completeness"] >= 0.7:
            report_lines.append(f"  ✓ 输出完整性好 ({metrics['avg_output_completeness']:.1%})")
        if metrics["avg_selling_points_quality"] >= 0.6:
            report_lines.append(f"  ✓ 卖点提取质量优秀 ({metrics['avg_selling_points_quality']:.1%})")
        if metrics["avg_latency"] <= 3.0:
            report_lines.append(f"  ✓ 响应速度快 ({metrics['avg_latency']:.2f}s)")

        report_lines.append("")
        report_lines.append("## 改进建议")
        report_lines.append("")

        if metrics["avg_category_accuracy"] < 0.8:
            report_lines.append("  • 类别识别准确率需要提升，建议微调模型或优化提示词")
        if metrics["avg_output_completeness"] < 0.7:
            report_lines.append("  • 输出完整性不足，建议在提示词中强调必需字段")
        if metrics["avg_selling_points_quality"] < 0.6:
            report_lines.append("  • 卖点提取质量有待提高，可以通过示例学习改进")
        if metrics["avg_format_validity"] < 0.8:
            report_lines.append("  • 输出格式不够规范，建议使用更严格的JSON输出约束")

    report_lines.append("")
    report_lines.append("=" * 80)

    # 打印报告
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告
    with open("/home/claude/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # 保存详细结果为JSON
    with open("/home/claude/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n详细结果已保存到:")
    print("  - /home/claude/evaluation_report.txt")
    print("  - /home/claude/evaluation_results.json")


if __name__ == "__main__":
    main()