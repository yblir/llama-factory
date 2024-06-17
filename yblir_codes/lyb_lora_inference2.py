# -*- coding: utf-8 -*-
# @Time    : 2024/5/16 23:50
# @Author  : yblir
# @File    : lyb_lora_inference.py
# explain  : 
# =======================================================
import yaml
import json
from loguru import logger
import time
import sys
from llamafactory.chat import ChatModel

if __name__ == '__main__':
    with open('../examples/yblir_configs/lyb_qwen_lora_merge_vllm.yaml', 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)

    chat_model = ChatModel(param)

    with open('/mnt/g/GoogleDownload/tuning_sample.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 预热
    messages = [{"role": "user", "content": data[0]['instruction']}]
    _ = chat_model.chat(messages)

    predict_1000 = []
    total_time = 0
    for i, item in enumerate(data):
        messages = [{"role": "user", "content": item['instruction']}]
        t1 = time.time()
        res = chat_model.chat(messages)
        total_time += time.time() - t1
        predict_1000.append(res[0].response_text)
        print(res[0].response_text)
        if (i + 1) % 100 == 0:
            logger.info(f'当前完成: {i + 1}')
            # sys.exit()
        if i + 1 == 300:
            break

    # json_data = json.dumps(predict_1000, indent=4, ensure_ascii=False)
    # with open('saves2/qwen_7b_chat_lora_merge_vllm.json', 'w', encoding='utf-8') as f:
    #     f.write(json_data)

    logger.success(f'写入完成, 总耗时:{total_time},平均耗时: {round((total_time / 300), 5)} s')
