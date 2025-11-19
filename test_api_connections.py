
"""
API连接测试脚本 (最终版)
包含: 阿里云 Qwen, Friendli 模型, HF BLIP-2
"""

import requests
import base64
import json
from io import BytesIO
from PIL import Image

# =============================================================================
# API配置
# =============================================================================

# 🔴 请在这里填入你的阿里云 API Key
DASHSCOPE_API_KEY = "sk-dbb9cb5041f641719b0daf6d4c0f67ed"

# 现有的 Token
HF_TOKEN = "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"
FRIENDLI_TOKEN = "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b"

# HF Endpoint (BLIP-2)
HF_BLIP2_ENDPOINT = "https://ib99q027z343t8rr.us-east-1.aws.endpoints.huggingface.cloud"

# 测试图片
TEST_IMAGE_URL = "https://m.media-amazon.com/images/I/51il3TLFnEL._AC_SL1000_.jpg"


def create_test_image_base64():
    """创建测试图片Base64"""
    try:
        print("⬇️  正在下载测试图片...")
        response = requests.get(TEST_IMAGE_URL, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.resize((512, 512)) # 缩小尺寸加快测试
        if img.mode != 'RGB': img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"✗ 图片准备失败: {e}")
        return None

def test_aliyun_qwen():
    """测试阿里云 Qwen-VL-Max"""
    print("\n" + "=" * 60)
    print("测试 1: Qwen-VL-Max (Alibaba Cloud DashScope)")
    print("=" * 60)

    if "sk-" not in DASHSCOPE_API_KEY:
        print("❌ 跳过: 请先在代码顶部填入有效的阿里云 DASHSCOPE_API_KEY")
        return

    # 阿里云兼容 OpenAI 格式的端点
    endpoint = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

    image_base64 = create_test_image_base64()
    if not image_base64: return

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-vl-max", # 使用通义千问VL Max版本
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this product strictly in English."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"✅ 成功! 响应: {response.json()['choices'][0]['message']['content']}")
        else:
            print(f"❌ 失败 ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ 错误: {e}")

def test_friendli_models():
    """测试 Friendli AI 模型 (InternVL & MiniCPM)"""
    models = [
        ("InternVL2_5-8B", "depx7m7tdfyych8"),
        ("MiniCPM-Llama3-V-2_5", "dep38olypw05c2m")
    ]

    endpoint = "https://api.friendli.ai/dedicated/v1/chat/completions"
    image_base64 = create_test_image_base64()
    if not image_base64: return

    headers = {
        "Authorization": f"Bearer {FRIENDLI_TOKEN}",
        "Content-Type": "application/json"
    }

    for name, model_id in models:
        print("\n" + "=" * 60)
        print(f"测试: {name} (Friendli AI)")
        print("=" * 60)

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 50
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                print(f"✅ 成功! 响应: {response.json()['choices'][0]['message']['content']}")
            else:
                print(f"❌ 失败 ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    test_aliyun_qwen()
    test_friendli_models()