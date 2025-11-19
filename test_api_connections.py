"""
API连接测试脚本
功能: 测试各个模型API的连接性和基本功能
"""

import requests
import base64
import json
from io import BytesIO
from PIL import Image

# API配置
HF_TOKEN = "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"
FRIENDLI_TOKEN = "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b"

# 测试用的简单图片URL
TEST_IMAGE_URL = "https://m.media-amazon.com/images/I/51il3TLFnEL._AC_SL1000_.jpg"


def create_test_image_base64():
    """创建一个测试图片的base64编码"""
    try:
        response = requests.get(TEST_IMAGE_URL, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # 调整大小
        img = img.resize((512, 512))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        print(f"✓ 测试图片已准备 (base64长度: {len(img_base64)} 字符)")
        return img_base64
    except Exception as e:
        print(f"✗ 准备测试图片失败: {e}")
        return None


def test_huggingface_qwen():
    """测试HuggingFace Qwen3-VL-32B端点"""
    print("\n" + "=" * 60)
    print("测试 1: Qwen3-VL-32B-Instruct (HuggingFace)")
    print("=" * 60)

    endpoint = "https://bfhfmuskpdnp4ajz.us-east-1.aws.endpoints.huggingface.cloud"

    image_base64 = create_test_image_base64()
    if not image_base64:
        return

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # 尝试多种payload格式
    print("\n尝试格式 1: 标准HF Inference格式...")
    payload1 = {
        "inputs": "Describe this product image briefly.",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.1
        }
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload1, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")
    except Exception as e:
        print(f"错误: {e}")

    print("\n尝试格式 2: 带图片的格式...")
    payload2 = {
        "inputs": {
            "question": "What product is in this image?",
            "image": image_base64
        }
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload2, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")
    except Exception as e:
        print(f"错误: {e}")

    print("\n尝试格式 3: Messages格式...")
    payload3 = {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "image": image_base64}
                    ]
                }
            ]
        },
        "parameters": {
            "max_tokens": 100
        }
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload3, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")
    except Exception as e:
        print(f"错误: {e}")


def test_friendli_internvl():
    """测试Friendli InternVL2.5-8B端点"""
    print("\n" + "=" * 60)
    print("测试 2: InternVL2_5-8B (Friendli AI)")
    print("=" * 60)

    endpoint = "https://api.friendli.ai/dedicated/v1/chat/completions"
    model_id = "depx7m7tdfyych8"

    image_base64 = create_test_image_base64()
    if not image_base64:
        return

    headers = {
        "Authorization": f"Bearer {FRIENDLI_TOKEN}",
        "Content-Type": "application/json"
    }

    print("\n尝试 OpenAI 兼容格式...")
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this product image in one sentence."
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
        "max_tokens": 100,
        "temperature": 0.1
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")

        if response.status_code == 200:
            data = response.json()
            if "choices" in data:
                print(f"\n✓ 成功! 模型输出: {data['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"错误: {e}")


def test_friendli_minicpm():
    """测试Friendli MiniCPM端点"""
    print("\n" + "=" * 60)
    print("测试 3: MiniCPM-Llama3-V-2_5 (Friendli AI)")
    print("=" * 60)

    endpoint = "https://api.friendli.ai/dedicated/v1/chat/completions"
    model_id = "dep38olypw05c2m"

    image_base64 = create_test_image_base64()
    if not image_base64:
        return

    headers = {
        "Authorization": f"Bearer {FRIENDLI_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is this product?"
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
        "max_tokens": 100
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")

        if response.status_code == 200:
            data = response.json()
            if "choices" in data:
                print(f"\n✓ 成功! 模型输出: {data['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"错误: {e}")


def test_huggingface_blip2():
    """测试HuggingFace BLIP2端点"""
    print("\n" + "=" * 60)
    print("测试 4: BLIP2-OPT-2.7B-NLI (HuggingFace)")
    print("=" * 60)

    endpoint = "https://onnunaok1tz3tbo2.us-east-1.aws.endpoints.huggingface.cloud"

    image_base64 = create_test_image_base64()
    if not image_base64:
        return

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # BLIP2通常用于图像描述生成
    print("\n尝试图像描述任务...")
    payload = {
        "inputs": image_base64  # BLIP2通常直接接受base64图片
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")

        if response.status_code == 200:
            print(f"\n✓ 成功!")
    except Exception as e:
        print(f"错误: {e}")


def test_api_health():
    """测试API端点健康状态"""
    print("\n" + "=" * 60)
    print("API端点健康检查")
    print("=" * 60)

    endpoints = {
        "HF Qwen3": "https://bfhfmuskpdnp4ajz.us-east-1.aws.endpoints.huggingface.cloud",
        "Friendli API": "https://api.friendli.ai/dedicated/v1/models",
        "HF BLIP2": "https://onnunaok1tz3tbo2.us-east-1.aws.endpoints.huggingface.cloud"
    }

    for name, url in endpoints.items():
        print(f"\n检查 {name}...")
        try:
            # 尝试HEAD请求
            response = requests.head(url, timeout=5)
            print(f"  HEAD状态码: {response.status_code}")
        except Exception as e:
            print(f"  HEAD请求失败: {e}")

        try:
            # 尝试GET请求
            response = requests.get(url, timeout=5)
            print(f"  GET状态码: {response.status_code}")
        except Exception as e:
            print(f"  GET请求失败: {e}")


if __name__ == "__main__":
    print("\n开始API连接测试...\n")

    # 首先检查端点健康状态
    test_api_health()

    # 测试各个模型
    test_huggingface_qwen()
    test_friendli_internvl()
    test_friendli_minicpm()
    test_huggingface_blip2()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)