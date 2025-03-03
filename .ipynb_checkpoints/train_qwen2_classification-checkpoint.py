import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # 指定单卡: 使用第一张卡


import pandas as pd
import json

from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import Dataset


train_jsonl_new_path = "data/zh_cls_fudan-news/new_train.jsonl"
test_jsonl_new_path = "data/zh_cls_fudan-news/new_test.jsonl"

# 加载训练 & 测试集
# train_df = pd.read_json(train_jsonl_new_path, lines=True)[:1000]  # 取前1000条做训练（可选）
train_df = pd.read_json(train_jsonl_new_path, lines=True)  # 取前1000条做训练（可选）

tokenizer = AutoTokenizer.from_pretrained('./qwen/Qwen2.5-1.5B-Instruct/', use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2.5-1.5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 预处理训练数据

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)


# 设置lora参数

from peft import LoraConfig, TaskType, get_peft_model
# from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# from transformers.models.qwen2 import Qwen2Model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)



# 训练
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

args = TrainingArguments(
    output_dir="./output/Qwen2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

from swanlab.integration.huggingface import SwanLabCallback
import swanlab

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune-classification",
    experiment_name="Qwen2.5-1.5B-Instruct-classification-lora",
    description="使用通义千问Qwen2.5-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
    config={
        "model": "qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "huangjintao/zh_cls_fudan-news",
    },   
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()


# ====== 训练结束后的预测 ===== #
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]  # 取前10条做主观评测

def predict(messages, model, tokenizer):
    device = "cuda:0"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    return response
    

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row["instruction"]
    input_value = row["input"]

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"},
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = json.dumps(messages, ensure_ascii=False)
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})