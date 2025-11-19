"""
两阶段感知层推理系统
作者: Shikang WANG

完整流程:
输入图片 → 阶段1: 轻量级优化器 → 优化Prompt → 阶段2: 最佳VLM → 高质量输出
"""

import json
import time
import base64
import requests
import numpy as np
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image
from pathlib import Path

# 导入优化器 (假设在同一目录)
from lightweight_prompt_optimizer import PromptOptimizer


class TwoStagePerceptionPipeline:
    """两阶段感知推理流水线"""

    def __init__(
            self,
            optimizer_path: str,
            best_model_key: str = "qwen3_vl_32b"
    ):
        """
        初始化两阶段流水线

        Args:
            optimizer_path: 优化器保存路径
            best_model_key: 最佳模型的标识符
        """
        print("\n🚀 初始化两阶段感知流水线...")

        # 阶段1: 加载轻量级优化器
        print("  [阶段1] 加载轻量级优化器...")
        self.optimizer = PromptOptimizer.load(optimizer_path)

        # 阶段2: 配置最佳 VLM
        print(f"  [阶段2] 配置最佳模型: {best_model_key}")
        self.best_model_key = best_model_key
        self.api_config = self._get_api_config(best_model_key)

        print("✅ 初始化完成!\n")

    def _get_api_config(self, model_key: str) -> Dict[str, Any]:
        """获取 API 配置"""
        configs = {
            "qwen3_vl_32b": {
                "name": "Qwen3-VL-32B-Instruct",
                "endpoint": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
                "token": "sk-dbb9cb5041f641719b0daf6d4c0f67ed",
                "model_id": "qwen-vl-max"
            },
            "internvl2_5_8b": {
                "name": "InternVL2.5-8B",
                "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
                "token": "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b",
                "model_id": "depx7m7tdfyych8"
            },
            "minicpm_llama3_v_2_5": {
                "name": "MiniCPM-Llama3-V-2.5",
                "endpoint": "https://api.friendli.ai/dedicated/v1/chat/completions",
                "token": "flp_2hxrOfWLYMxzXe5h58NuOJ4IE1Yo6z2KfZ3xwe4RdFR06b",
                "model_id": "dep38olypw05c2m"
            }
        }
        return configs[model_key]

    def extract_image_features(self, image_url: str) -> Optional[np.ndarray]:
        """
        提取图像特征 (简化版本)

        实际应用中应使用:
        - CLIP: OpenAI的视觉-语言模型
        - ResNet: 图像分类backbone
        - DINO: 自监督视觉特征

        这里使用图像统计特征作为演示
        """
        try:
            response = requests.get(image_url, timeout=15)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = img.resize((224, 224))

            # 简单的颜色直方图特征
            img_array = np.array(img)

            # RGB通道均值和标准差
            color_features = []
            for channel in range(3):
                color_features.append(img_array[:, :, channel].mean())
                color_features.append(img_array[:, :, channel].std())

            # 纹理特征 (梯度)
            gray = np.mean(img_array, axis=2)
            grad_x = np.abs(np.diff(gray, axis=1)).mean()
            grad_y = np.abs(np.diff(gray, axis=0)).mean()
            color_features.extend([grad_x, grad_y])

            # 填充到512维 (与训练时一致)
            features = np.array(color_features + [0] * (512 - len(color_features)), dtype=np.float32)

            return features

        except Exception as e:
            print(f"⚠️  特征提取失败: {e}")
            return None

    def download_and_encode_image(self, image_url: str) -> Optional[str]:
        """下载并编码图片"""
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            if max(img.size) > 1024:
                ratio = 1024 / max(img.size)
                img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"❌ 图片处理失败: {e}")
            return None

    def call_vlm_api(self, image_base64: str, prompt: str) -> tuple[str, float, str]:
        """调用 VLM API"""
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_config['token']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.api_config["model_id"],
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

        try:
            response = requests.post(
                self.api_config["endpoint"],
                headers=headers,
                json=payload,
                timeout=30
            )

            latency = time.time() - start_time

            if response.status_code != 200:
                return "", latency, f"API Error {response.status_code}"

            content = response.json()["choices"][0]["message"]["content"]
            return content, latency, ""

        except Exception as e:
            return "", time.time() - start_time, str(e)

    def run(
            self,
            image_url: str,
            category_hint: Optional[str] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整的两阶段推理

        Args:
            image_url: 产品图片URL
            category_hint: 可选的类别提示 (如果不提供则自动预测)
            verbose: 是否打印详细信息

        Returns:
            包含结果的字典
        """
        if verbose:
            print("\n" + "=" * 80)
            print("🎯 两阶段感知推理")
            print("=" * 80)

        result = {
            "image_url": image_url,
            "stage1_time": 0.0,
            "stage2_time": 0.0,
            "total_time": 0.0,
            "success": False,
            "error": "",
            "optimized_prompt": "",
            "raw_output": "",
            "parsed_attributes": {}
        }

        total_start = time.time()

        # ========== 阶段1: 轻量级优化器 ==========
        if verbose:
            print("\n[阶段1] 轻量级优化器处理")

        stage1_start = time.time()

        # 1.1 提取图像特征
        if verbose:
            print("  📊 提取图像特征...")
        image_features = self.extract_image_features(image_url)

        # 1.2 生成优化 Prompt
        if verbose:
            print("  🔧 生成优化 Prompt...")

        if category_hint:
            if verbose:
                print(f"  💡 使用提供的类别提示: {category_hint}")
            optimized_prompt_obj = self.optimizer.optimize_prompt(category_hint=category_hint)
        elif image_features is not None:
            optimized_prompt_obj = self.optimizer.optimize_prompt(image_features=image_features)
        else:
            if verbose:
                print("  ⚠️  特征提取失败，使用基础 Prompt")
            optimized_prompt_obj = self.optimizer.optimize_prompt()

        optimized_prompt = optimized_prompt_obj.to_full_prompt()
        result["optimized_prompt"] = optimized_prompt

        stage1_time = time.time() - stage1_start
        result["stage1_time"] = stage1_time

        if verbose:
            print(f"  ✅ 阶段1完成 (耗时: {stage1_time:.2f}s)")

        # ========== 阶段2: 最佳 VLM 推理 ==========
        if verbose:
            print(f"\n[阶段2] {self.api_config['name']} 推理")

        stage2_start = time.time()

        # 2.1 准备图片
        if verbose:
            print("  📸 准备图片...")
        image_base64 = self.download_and_encode_image(image_url)

        if not image_base64:
            result["error"] = "图片下载失败"
            result["total_time"] = time.time() - total_start
            return result

        # 2.2 调用 VLM API
        if verbose:
            print("  🤖 调用 VLM API...")

        raw_output, api_latency, error = self.call_vlm_api(image_base64, optimized_prompt)

        stage2_time = time.time() - stage2_start
        result["stage2_time"] = stage2_time
        result["raw_output"] = raw_output

        if error:
            result["error"] = error
            result["total_time"] = time.time() - total_start
            if verbose:
                print(f"  ❌ API调用失败: {error}")
            return result

        # 2.3 解析输出
        if verbose:
            print("  📝 解析输出...")

        try:
            # 移除可能的 Markdown 格式
            import re
            cleaned = re.sub(r'```(?:json)?\s*', '', raw_output)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            attributes = json.loads(cleaned.strip())
            result["parsed_attributes"] = attributes
            result["success"] = True

        except json.JSONDecodeError as e:
            result["error"] = f"JSON解析失败: {e}"
            if verbose:
                print(f"  ⚠️  JSON解析失败，尝试启发式解析...")

            # 启发式解析 (backup)
            from lightweight_prompt_optimizer import extract_json_from_text
            attributes = extract_json_from_text(raw_output)

            if attributes:
                result["parsed_attributes"] = attributes
                result["success"] = True
                if verbose:
                    print("  ✅ 启发式解析成功")

        result["total_time"] = time.time() - total_start

        if verbose:
            print(f"  ✅ 阶段2完成 (耗时: {stage2_time:.2f}s)")
            print(f"\n{'=' * 80}")
            print(f"✅ 推理完成!")
            print(f"{'=' * 80}")
            print(f"  阶段1耗时: {stage1_time:.2f}s")
            print(f"  阶段2耗时: {stage2_time:.2f}s")
            print(f"  总耗时:     {result['total_time']:.2f}s")

            if result["success"]:
                attrs = result["parsed_attributes"]
                print(f"\n📦 提取的属性:")
                print(f"  - 类别:     {attrs.get('category', 'N/A')}")
                print(f"  - 产品:     {attrs.get('main_product', 'N/A')}")
                print(f"  - 颜色:     {attrs.get('color', 'N/A')}")
                print(f"  - 材质:     {attrs.get('material', 'N/A')}")
                print(f"  - 卖点数:   {len(attrs.get('selling_points', []))}")

        return result

    def batch_run(
            self,
            image_urls: list[str],
            save_path: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """批量处理"""
        print(f"\n🔄 批量处理 {len(image_urls)} 张图片...")

        results = []
        for idx, url in enumerate(image_urls, 1):
            print(f"\n处理 {idx}/{len(image_urls)}")
            result = self.run(url, verbose=False)
            results.append(result)

            # 显示简要结果
            if result["success"]:
                print(f"  ✅ 成功 (总耗时: {result['total_time']:.2f}s)")
            else:
                print(f"  ❌ 失败: {result['error']}")

            # API限流
            if idx < len(image_urls):
                time.sleep(1.5)

        # 保存结果
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存至: {save_path}")

        # 统计
        successful = sum(1 for r in results if r["success"])
        print(f"\n📊 批量处理完成:")
        print(f"  成功: {successful}/{len(results)}")
        print(f"  平均耗时: {np.mean([r['total_time'] for r in results]):.2f}s")

        return results


# ==================== 使用示例 ====================

def demo():
    """演示两阶段推理系统"""

    print("\n" + "=" * 80)
    print("🎭 两阶段感知系统演示")
    print("=" * 80)

    # 1. 初始化系统
    pipeline = TwoStagePerceptionPipeline(
        optimizer_path="trained_optimizer",
        best_model_key="qwen3_vl_32b"  # 根据评估结果选择
    )

    # 2. 单张测试
    test_url = "https://m.media-amazon.com/images/I/51il3TLFnEL._AC_SL1000_.jpg"

    result = pipeline.run(test_url, verbose=True)

    # 3. 显示结果
    if result["success"]:
        print("\n" + "=" * 80)
        print("📄 完整结果")
        print("=" * 80)
        print(json.dumps(result["parsed_attributes"], indent=2, ensure_ascii=False))

    # 4. 批量测试 (可选)
    # test_urls = [url1, url2, url3, ...]
    # pipeline.batch_run(test_urls, save_path="batch_results.json")


if __name__ == "__main__":
    demo()