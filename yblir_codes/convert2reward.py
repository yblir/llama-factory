# -*- coding: utf-8 -*-
# @Time    : 2024/6/25 上午9:01
# @Author  : yblir
# @File    : convert2reward.py
# explain  : 
# =======================================================
import json

if __name__ == '__main__':
    with open('../data/rw_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    for item in data:
        temp = {}
        temp['conversations'] = [{'from': 'human', 'value': item['instruction']}, ]
        temp['chosen'] = {'from': 'gpt', 'value': item['chosen']}
        temp['rejected'] = {'from': 'gpt', 'value': item['rejected']}
        new_data.append(temp)

    with open('../data/rw_data2.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_data, indent=4, ensure_ascii=False))
