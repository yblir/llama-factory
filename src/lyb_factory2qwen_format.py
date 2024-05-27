# -*- coding: utf-8 -*-
# @Time    : 2024/5/26 10:47
# @Author  : yblir
# @File    : lyb_factory2qwen_format.py
# explain  : 
# =======================================================
import json

# [
#   {
#     "id": "identity_0",
#     "conversations": [
#       {
#         "from": "user",
#         "value": "你好"
#       },
#       {
#         "from": "assistant",
#         "value": "我是一个语言模型，我叫通义千问。"
#       }
#     ]
#   }
# ]
qwen_format_list = []
with open('/mnt/e/PyCharm/insteresting/LLaMA-Factory-0.7.1/data/qwen-7b-sql-gptq.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, item in enumerate(data):
    temp1 = {}
    temp1['id'] = f'identity_{i}'
    temp1['conversations'] = [{'from': 'user', 'value': item['instruction']},
                              {'from': 'assistant', 'value': item['output']}]

    qwen_format_list.append(temp1)

with open('/mnt/e/PyCharm/insteresting/LLaMA-Factory-0.7.1/data/qwen-7b-sql-gptq2.json', 'w', encoding='utf-8') as f:
    json.dump(qwen_format_list, f, indent=4, ensure_ascii=False)
