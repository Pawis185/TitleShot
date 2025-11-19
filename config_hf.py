# config_hf.py
# 全局配置 + 公共常量（中文注释）

from typing import List, Dict

# 数据集中的所有类别（和 dataset.json 里的 key 对应）
CATEGORIES: List[str] = [
    "Automotive",
    "Books",
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Movies_and_TV",
    "Office_Products",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]

# 在 Hugging Face 上对应的模型仓库名
# ✅ 注意：如果你的实际模型 repo 名有差异，请在这里改
CANDIDATE_MODELS: Dict[str, str] = {
    "qwen2_vl_7b": "Qwen/Qwen2-VL-7B-Instruct",
    "internvl2_8b": "OpenGVLab/InternVL2-8B",
    "florence2_large": "microsoft/Florence-2-large",
    "minicpm_v_2_5": "openbmb/MiniCPM-V-2_5",  # 有些版本叫 MiniCPM-V-2_6，注意对照 Hugging Face 上名字
    "blip2_opt_2_7b": "Salesforce/blip2-opt-2.7b",
}

# 统一的分类提示词（会作为 user 的 text content）
# 让模型**只输出一个类别标签**
CLASSIFICATION_INSTRUCTION = (
    "请根据图片内容，从下列类别中选择一个最合适的类别：\n"
    "{categories}\n\n"
    "要求：\n"
    "1. 只输出**一个**类别标签；\n"
    "2. 输出的类别标签必须和列表中的标签完全一致（区分大小写）；\n"
    "3. 不要输出额外解释或标点。"
)
