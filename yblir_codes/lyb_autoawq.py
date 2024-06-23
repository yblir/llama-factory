# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午10:55
# @Author  : yblir
# @File    : lyb_autoawq.py
# explain  : 
# =======================================================
import json
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def qwen_preprocess(lora_data_, tokenizer_, max_len_):
    """
    最终处理后，msg格式如下，system要改成自己的：
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ]
    """

    messages = []
    for item in lora_data_:
        temp = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item['instruction']},
            {"role": "assistant", "content": item['output']}
        ]
        messages.append(temp)

    data = []
    for msg in messages:
        text = tokenizer_.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        data.append(text)

    return data


if __name__ == '__main__':
    # Specify paths and hyperparameters for quantization
    model_dir_path = "/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b_merge_800"
    quantized_path = "/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b_merge_800_int4_awq"
    # 验证prompt的单条数据
    # quantize_dataset_path = '/media/xk/D6B8A862B8A8433B/GitHub/llama-factory/data/train_clean_test.json'
    quantize_dataset_path = '/media/xk/D6B8A862B8A8433B/GitHub/llama-factory/data/train_clean.json'

    with open(quantize_dataset_path, 'r', encoding='utf-8') as f:
        lora_data = json.load(f)

    max_len = 2048
    # GEMM:文本长或batch_size比较大时，速度会快，少文本时，GEMV会更快
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # Load your tokenizer and model with AutoAWQ
    tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
    model = AutoAWQForCausalLM.from_pretrained(model_dir_path, device_map="auto", safetensors=True)

    data = qwen_preprocess(lora_data, tokenizer, max_len)

    # 默认数据长度大于512，该条数据不使用
    model.quantize(tokenizer, quant_config=quant_config, calib_data=data)

    model.save_quantized(quantized_path, safetensors=True)
    tokenizer.save_pretrained(quantized_path)
