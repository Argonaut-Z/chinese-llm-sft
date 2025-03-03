---
tasks:
- text-generation
- chat
license: Apache License 2.0
#用户自定义标签
tags:
- text-classification
- text-generation
- chat
- zh
configs:
  - config_name: default
    data_files:
      - split: train
        path: "train.jsonl"
  - config_name: test
    data_files:
      - split: test
        path: "test.jsonl"  
---


该任务属于开放域的分类问题

下载: 
```python
from modelscope import MsDataset
dataset = MsDataset.load('swift/zh_cls_fudan-news', split='train')
test_dataset = MsDataset.load('swift/zh_cls_fudan-news', subset_name='test', split='test')
print(dataset)
print(test_dataset)
"""
Dataset({
    features: ['text', 'category', 'output'],
    num_rows: 4000
})
Dataset({
    features: ['text', 'category', 'output'],
    num_rows: 959
})
"""
```


