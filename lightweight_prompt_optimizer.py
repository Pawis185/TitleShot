"""
轻量级 Prompt 优化器
作者: Shikang WANG
功能: 训练小型模型优化 VLM 输入，提升最终输出质量

两阶段架构:
阶段1: 轻量级优化模型 (本地部署)
    - 属性分类器: 预测产品类别，生成针对性提示
    - 卖点提取器: 识别关键特征，强化 prompt
    - 格式规范器: 确保输出标准化

阶段2: 最佳 VLM 模型 (API 调用)
    - 基于优化后的 prompt 生成高质量输出
"""

import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle


# ==================== 数据结构 ====================

@dataclass
class OptimizedPrompt:
    """优化后的 Prompt"""
    base_prompt: str
    category_hint: str
    feature_emphasis: List[str]
    output_constraints: Dict[str, str]

    def to_full_prompt(self) -> str:
        """生成完整的优化 prompt"""
        prompt = self.base_prompt + "\n\n"
        prompt += f"IMPORTANT CONTEXT:\n"
        prompt += f"- Product Category: {self.category_hint}\n"

        if self.feature_emphasis:
            prompt += f"- Key Features to Highlight: {', '.join(self.feature_emphasis)}\n"

        prompt += f"\nOUTPUT REQUIREMENTS:\n"
        for field, requirement in self.output_constraints.items():
            prompt += f"- {field}: {requirement}\n"

        return prompt


# ==================== 属性分类器 ====================

class CategoryClassifier(nn.Module):
    """轻量级类别分类器 (基于视觉特征)"""

    def __init__(self, num_categories: int, feature_dim: int = 512):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_categories)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(features)


class CategoryPredictor:
    """类别预测器 (训练和推理)"""

    def __init__(self, categories: List[str]):
        self.categories = categories
        self.num_categories = len(categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}

        self.model = CategoryClassifier(self.num_categories)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_model(
            self,
            train_features: np.ndarray,
            train_labels: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            learning_rate: float = 0.001
    ):
        """训练分类器"""
        print("\n🎓 训练类别分类器...")

        # 转换为 tensor
        X = torch.FloatTensor(train_features).to(self.device)
        y = torch.LongTensor(train_labels).to(self.device)

        # 数据集
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_labels).sum().item()
                total += batch_labels.size(0)

            if (epoch + 1) % 10 == 0:
                accuracy = 100. * correct / total
                print(f"  Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}, Acc: {accuracy:.2f}%")

        print("✅ 训练完成!")

    def predict(self, features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """预测类别 (返回 top-k 及其概率)"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 获取 top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        results = [(self.idx_to_category[idx], float(probs[idx])) for idx in top_indices]

        return results

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'categories': self.categories,
            'category_to_idx': self.category_to_idx
        }, path)
        print(f"💾 分类器已保存至: {path}")

    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        predictor = cls(checkpoint['categories'])
        predictor.model.load_state_dict(checkpoint['model_state'])
        return predictor


# ==================== 特征提取规则引擎 ====================

class FeatureExtractor:
    """基于规则的特征提取 (无需训练)"""

    # 类别相关的关键特征
    CATEGORY_FEATURES = {
        "Electronics": ["connectivity", "battery life", "display", "processor", "storage"],
        "Clothing": ["material", "fit", "comfort", "style", "durability"],
        "Home & Kitchen": ["capacity", "material", "ease of cleaning", "design", "versatility"],
        "Books": ["genre", "author", "length", "edition", "format"],
        "Sports & Outdoors": ["durability", "weather resistance", "portability", "performance"],
        "Automotive": ["compatibility", "installation", "durability", "performance"],
        "Tools & Home Improvement": ["power", "precision", "durability", "safety features"],
        "Pet Supplies": ["safety", "comfort", "durability", "easy to clean"],
        "Health & Personal Care": ["ingredients", "effectiveness", "gentleness", "scent"],
        "Toys & Games": ["age suitability", "educational value", "safety", "entertainment"],
    }

    # 通用卖点关键词
    SELLING_POINT_KEYWORDS = [
        "premium", "durable", "high-quality", "affordable", "versatile",
        "easy to use", "comfortable", "stylish", "eco-friendly", "safe",
        "powerful", "efficient", "reliable", "innovative", "bestselling"
    ]

    def extract_features_for_category(self, category: str, top_k: int = 3) -> List[str]:
        """根据类别提取关键特征"""
        # 标准化类别名称
        category_normalized = category.replace("_", " ").replace("&", "and").strip()

        # 查找匹配的类别
        for cat_key, features in self.CATEGORY_FEATURES.items():
            if cat_key.lower() in category_normalized.lower() or \
                    category_normalized.lower() in cat_key.lower():
                return features[:top_k]

        # 默认通用特征
        return ["quality", "design", "value"]

    def generate_selling_point_hints(self, category: str) -> List[str]:
        """生成卖点提示"""
        # 选择与类别相关的卖点关键词
        hints = []

        if "electronics" in category.lower():
            hints = ["advanced technology", "high performance", "long battery life"]
        elif "clothing" in category.lower() or "jewelry" in category.lower():
            hints = ["premium materials", "comfortable fit", "stylish design"]
        elif "home" in category.lower() or "kitchen" in category.lower():
            hints = ["practical design", "easy to clean", "space-saving"]
        elif "sports" in category.lower():
            hints = ["durable construction", "weather resistant", "lightweight"]
        else:
            hints = ["high quality", "great value", "customer favorite"]

        return hints


# ==================== 格式规范器 ====================

class FormatValidator:
    """输出格式验证和修正"""

    REQUIRED_FIELDS = [
        "category", "main_product", "color", "material", "style",
        "key_features", "selling_points", "target_audience", "usage_scenario"
    ]

    FIELD_CONSTRAINTS = {
        "category": {"type": str, "min_length": 2},
        "main_product": {"type": str, "min_length": 3},
        "color": {"type": str, "min_length": 2},
        "material": {"type": str, "min_length": 2},
        "style": {"type": str, "min_length": 2},
        "key_features": {"type": list, "min_items": 2, "max_items": 5},
        "selling_points": {"type": list, "min_items": 2, "max_items": 3},
        "target_audience": {"type": str, "min_length": 3},
        "usage_scenario": {"type": list, "min_items": 1, "max_items": 3}
    }

    def generate_constraints_text(self) -> Dict[str, str]:
        """生成约束文本 (用于 prompt)"""
        constraints = {}
        for field, rules in self.FIELD_CONSTRAINTS.items():
            if rules["type"] == list:
                constraints[field] = f"Array with {rules['min_items']}-{rules['max_items']} items"
            else:
                constraints[field] = f"String with minimum {rules['min_length']} characters"
        return constraints

    def validate_output(self, output: Dict) -> Tuple[bool, List[str]]:
        """验证输出是否符合规范"""
        errors = []

        # 检查必需字段
        for field in self.REQUIRED_FIELDS:
            if field not in output:
                errors.append(f"Missing field: {field}")
                continue

            value = output[field]
            rules = self.FIELD_CONSTRAINTS[field]

            # 类型检查
            if not isinstance(value, rules["type"]):
                errors.append(f"{field}: Expected {rules['type'].__name__}, got {type(value).__name__}")
                continue

            # 长度/数量检查
            if rules["type"] == str:
                if len(value) < rules["min_length"]:
                    errors.append(f"{field}: Too short (min {rules['min_length']} chars)")
            elif rules["type"] == list:
                if len(value) < rules["min_items"]:
                    errors.append(f"{field}: Too few items (min {rules['min_items']})")
                elif len(value) > rules["max_items"]:
                    errors.append(f"{field}: Too many items (max {rules['max_items']})")

        return len(errors) == 0, errors


# ==================== Prompt 优化器主类 ====================

class PromptOptimizer:
    """轻量级 Prompt 优化器 (整合所有组件)"""

    def __init__(self, categories: List[str]):
        self.categories = categories
        self.category_predictor = CategoryPredictor(categories)
        self.feature_extractor = FeatureExtractor()
        self.format_validator = FormatValidator()

        # 基础 prompt 模板
        self.base_prompt = """You are an expert e-commerce product analyst. Analyze this product image and extract key attributes.

Respond ONLY with valid JSON in this EXACT format (no additional text):
{
  "category": "primary product category",
  "main_product": "specific product name",
  "color": "primary color(s)",
  "material": "main material",
  "style": "design style",
  "key_features": ["feature1", "feature2", "feature3"],
  "selling_points": ["unique selling point 1", "unique selling point 2"],
  "target_audience": "target customer segment",
  "usage_scenario": ["scenario1", "scenario2"]
}"""

    def train_category_predictor(
            self,
            features: np.ndarray,
            labels: List[str],
            **train_kwargs
    ):
        """训练类别预测器"""
        # 转换标签为索引
        label_indices = np.array([
            self.category_predictor.category_to_idx[label]
            for label in labels
        ])

        self.category_predictor.train_model(features, label_indices, **train_kwargs)

    def optimize_prompt(
            self,
            image_features: Optional[np.ndarray] = None,
            category_hint: Optional[str] = None
    ) -> OptimizedPrompt:
        """生成优化的 prompt"""

        # 1. 预测类别 (如果提供特征)
        if image_features is not None and category_hint is None:
            predictions = self.category_predictor.predict(image_features, top_k=1)
            category_hint = predictions[0][0]
            confidence = predictions[0][1]
            print(f"  🔮 预测类别: {category_hint} (置信度: {confidence:.2%})")

        if category_hint is None:
            category_hint = "General Product"

        # 2. 提取特征强调项
        feature_emphasis = self.feature_extractor.extract_features_for_category(category_hint)
        selling_hints = self.feature_extractor.generate_selling_point_hints(category_hint)

        # 3. 生成格式约束
        output_constraints = self.format_validator.generate_constraints_text()

        # 4. 构建优化 prompt
        optimized = OptimizedPrompt(
            base_prompt=self.base_prompt,
            category_hint=category_hint,
            feature_emphasis=feature_emphasis + selling_hints,
            output_constraints=output_constraints
        )

        return optimized

    def save(self, save_dir: str):
        """保存优化器"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存类别预测器
        self.category_predictor.save(str(save_path / "category_predictor.pth"))

        # 保存配置
        config = {
            "categories": self.categories,
            "base_prompt": self.base_prompt
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 优化器已保存至: {save_dir}")

    @classmethod
    def load(cls, save_dir: str):
        """加载优化器"""
        save_path = Path(save_dir)

        # 加载配置
        with open(save_path / "config.json", 'r') as f:
            config = json.load(f)

        # 创建实例
        optimizer = cls(config["categories"])
        optimizer.base_prompt = config["base_prompt"]

        # 加载类别预测器
        optimizer.category_predictor = CategoryPredictor.load(
            str(save_path / "category_predictor.pth")
        )

        print(f"✅ 优化器已从 {save_dir} 加载")
        return optimizer


# ==================== 训练脚本 ====================

def train_optimizer_from_evaluation_results(
        evaluation_results_path: str = "evaluation_results.json",
        output_dir: str = "trained_optimizer"
) -> PromptOptimizer:
    """从评估结果训练优化器"""

    print("\n" + "=" * 80)
    print("🎓 训练轻量级 Prompt 优化器")
    print("=" * 80)

    # 1. 加载评估结果
    print("\n[1/3] 加载评估结果...")
    with open(evaluation_results_path, 'r') as f:
        eval_results = json.load(f)

    # 找到最佳模型的结果
    best_model_key = max(
        eval_results.keys(),
        key=lambda k: eval_results[k]["summary"]["overall_score"]
    )
    best_results = eval_results[best_model_key]["detailed_results"]

    print(f"✅ 使用最佳模型 {eval_results[best_model_key]['model_name']} 的结果")
    print(f"   共 {len(best_results)} 个样本")

    # 2. 准备训练数据
    print("\n[2/3] 准备训练数据...")

    # 提取类别
    categories = list(set([r["ground_truth"] for r in best_results]))
    print(f"✅ 发现 {len(categories)} 个类别: {categories}")

    # 模拟特征 (实际应用中应使用真实的图像特征)
    # 这里我们使用随机特征作为示例
    print("⚠️  注意: 使用模拟特征。实际应用需要提取真实图像特征 (如 CLIP embeddings)")

    features = []
    labels = []
    for result in best_results:
        # 模拟512维特征向量
        feature = np.random.randn(512).astype(np.float32)
        features.append(feature)
        labels.append(result["ground_truth"])

    features = np.array(features)

    # 3. 训练优化器
    print("\n[3/3] 训练优化器...")
    optimizer = PromptOptimizer(categories)

    optimizer.train_category_predictor(
        features=features,
        labels=labels,
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )

    # 4. 保存
    optimizer.save(output_dir)

    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print(f"\n使用方法:")
    print(f"  optimizer = PromptOptimizer.load('{output_dir}')")
    print(f"  optimized_prompt = optimizer.optimize_prompt(image_features)")

    return optimizer


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 训练优化器
    optimizer = train_optimizer_from_evaluation_results()

    # 示例: 使用优化器
    print("\n" + "=" * 80)
    print("🎯 使用示例")
    print("=" * 80)

    # 模拟图像特征
    sample_features = np.random.randn(512).astype(np.float32)

    # 生成优化 prompt
    optimized = optimizer.optimize_prompt(sample_features)

    print("\n📝 优化后的 Prompt:")
    print("-" * 80)
    print(optimized.to_full_prompt())
    print("-" * 80)