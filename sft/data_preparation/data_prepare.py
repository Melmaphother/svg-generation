import json
import random
from datasets import load_dataset

QWEN_2_5_VL_3B_Text_2_SVG_V1 = """
You are a helpful assistant assisting researchers in generating SVG code from textual descriptions. You will be provided with details to guide your SVG creation. Your task is to write SVG code that accurately represents the given textual information to the fullest extent possible. You are committed to solving the task of SVG generation for a robust system, so always strive to produce the best SVG code you can. Feel free to use multiple paths and any necessary shapes, colors, or lines to generate compilable SVG code within a maximum of 8000 tokens. The goal is to ensure the resulting SVG, when rasterized, best represents the described content. Respond only with the SVG code, enclosed in triple quotes, that directly corresponds to the provided textual description. Generate SVG code ONLY. Output a complete, well-formed SVG document directly, ending with </svg>. Do NOT include any Markdown fences (e.g., ```svg ```), additional text, explanations, or extraneous content whatsoever.
"""

def convert_dataset():
    # 读取原始数据集
    dataset = load_dataset(
        "/data/wdy/StarVector/text2svg-stack",
        split="train",
    )
    # 获取 dataset 数量
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    dataset = dataset.select(range(2000))
    
    # 创建新格式的数据
    new_data = []

    # 遍历每个样本
    for idx, item in enumerate(dataset):
        # print(idx, type(item), item)
        system_prompt = QWEN_2_5_VL_3B_Text_2_SVG_V1
        user_prompt = item['caption_cogvlm']
        model_response = item['Svg']

        # 创建新格式的样本，加入分类提示语
        new_sample = {
            "instruction": system_prompt,
            "input": user_prompt,
            "output": model_response
        }

        new_data.append(new_sample)

    # 随机打乱数据
    random.shuffle(new_data)
    
    # 计算训练集大小（90%的数据）
    train_size = int(len(new_data) * 1)
    
    # 划分训练集和测试集
    train_data = new_data[:train_size]
    test_data = new_data[train_size:]
    
    # 保存训练集
    with open('sft_data_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    with open('sft_data_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 打印数据集大小信息
    print(f"Total samples: {len(new_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    convert_dataset()