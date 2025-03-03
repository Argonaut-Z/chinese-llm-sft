# 将train.jsonl和test.jsonl进行处理，转换成new_train.jsonl和new_test.jsonl

import json
import pandas as pd
import os

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            context = data["text"]
            catagory = data["category"]
            label = data["output"]
            message = {
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本:{context},类型选型:{catagory}",
                "output": label,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 加载、处理数据集和测试集
train_dataset_path = "data/zh_cls_fudan-news/train.jsonl"
test_dataset_path = "data/zh_cls_fudan-news/test.jsonl"

train_jsonl_new_path = "data/zh_cls_fudan-news/new_train.jsonl"
test_jsonl_new_path = "data/zh_cls_fudan-news/new_test.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

