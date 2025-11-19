"""
电商产品图像感知层模型评估系统
功能: 评估多个视觉-语言模型 (Aliyun Qwen, Friendli Models, HF BLIP2)
"""

import json
import time
import base64
import requests
import numpy as np
from typing import Dict, List, Any, Tuple
from io import BytesIO
from PIL import Image
import re

# ==================== 配置部分 ====================

# 🔴 请在这里填入你的阿里云 API Key
DASHSCOPE_API_KEY = "sk-dbb9cb5041f641719b0daf6d4c0f67ed"

HF_TOKEN = "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"
FRIENDLI_TOKEN = "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b"

# 模型配置
MODEL_CONFIGS = {
    # 1. 阿里云 Qwen (替代了原有的 HF Qwen)
    "qwen_vl_max": {
        "name": "Qwen-VL-Max (Alibaba)",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_type": "aliyun",
        "token": DASHSCOPE_API_KEY,
        "model_id": "qwen-vl-max"
    },
    # 2. Friendli InternVL
    "internvl2_5_8b": {
        "name": "InternVL2_5-8B",
        "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
        "api_type": "friendli",
        "token": FRIENDLI_TOKEN,
        "model_id": "depx7m7tdfyych8"
    },
    # 3. Friendli MiniCPM
    "minicpm_llama3_v_2_5": {
        "name": "MiniCPM-Llama3-V-2_5",
        "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
        "api_type": "friendli",
        "token": FRIENDLI_TOKEN,
        "model_id": "dep38olypw05c2m"
    }
}

# 提示词
ATTRIBUTE_EXTRACTION_PROMPT = """Analyze this product image and extract the following information in JSON format:
{
  "category": "product category",
  "main_product": "product name",
  "color": "color list",
  "material": "material",
  "key_features": ["feature1", "feature2"],
  "selling_points": ["point1", "point2"],
  "target_audience": "audience",
  "usage_scenario": ["scenario1"]
}
Output JSON only."""


# ==================== 工具函数 ====================

def download_and_encode_image(image_url: str) -> Tuple[str, bool]:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)))
        if img.mode != 'RGB': img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8'), True
    except Exception as e:
        print(f"[警告] 图片下载失败: {e}")
        return "", False

def call_openai_style_endpoint(endpoint, token, model_id, image_base64, prompt, timeout=30):
    """通用的 OpenAI 格式调用 (适用于 Aliyun 和 Friendli)"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()

def extract_json(text):
    if not text: return {}
    try: return json.loads(text)
    except: pass
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.I)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return {}

def call_model_api(model_key, image_base64, prompt):
    config = MODEL_CONFIGS[model_key]
    start = time.time()

    try:
        # 1. Aliyun 和 Friendli 都兼容 OpenAI 格式，可以共用逻辑
        if config["api_type"] in ["aliyun", "friendli"]:
            resp = call_openai_style_endpoint(
                config["endpoint"], config["token"], config["model_id"], image_base64, prompt
            )
            content = resp["choices"][0]["message"]["content"]

        else:
            return {}, 0, "未知API类型"

        attrs = extract_json(content)
        return attrs, time.time() - start, ""

    except Exception as e:
        return {}, time.time() - start, str(e)

# ==================== 评估核心逻辑 ====================
# (这部分保持原样，只是为了完整性)

def evaluate_model(model_key, test_samples):
    print(f"\n评估模型: {MODEL_CONFIGS[model_key]['name']}")
    results = []
    for i, sample in enumerate(test_samples, 1):
        print(f"  样本 {i}...", end="", flush=True)
        img_b64, ok = download_and_encode_image(sample["image_url"])
        if not ok: continue

        attrs, lat, err = call_model_api(model_key, img_b64, ATTRIBUTE_EXTRACTION_PROMPT)

        res = {
            "success": not bool(err),
            "latency": lat,
            "completeness": len([k for k in attrs if attrs[k]]) / 8 if attrs else 0
        }
        results.append(res)
        print(f" {'✓' if res['success'] else '✗'} (延迟: {lat:.2f}s)")
        if i < len(test_samples): time.sleep(1)

    success_list = [r for r in results if r["success"]]
    score = np.mean([r["completeness"] for r in success_list]) if success_list else 0
    print(f"  完成! 平均完整度得分: {score:.2%}")
    return results

def main():
    print("开始评估...")
    # 这里使用模拟数据，实际请加载你的 dataset.json
    test_samples = [{"category": "Electronics", "image_url": "https://m.media-amazon.com/images/I/51il3TLFnEL._AC_SL1000_.jpg"}]

    for key in MODEL_CONFIGS:
        if key == "qwen_vl_max" and "sk-" not in DASHSCOPE_API_KEY:
            print(f"\n跳过 {key}: 未配置阿里云 API Key")
            continue
        try:
            evaluate_model(key, test_samples)
        except Exception as e:
            print(f"模型 {key} 评估出错: {e}")

if __name__ == "__main__":
    main()