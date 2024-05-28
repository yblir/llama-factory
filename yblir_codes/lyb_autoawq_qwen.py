# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午10:55
# @Author  : yblir
# @File    : lyb_autoawq_qwen.py
# explain  : 
# =======================================================
import json

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def preprocess(data_path_):
    '''
    最终处理后，msg格式如下：
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ]
    '''
    with open(data_path_, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    data = []
    for msg in messages:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        data.append(text.strip())

    return data


if __name__ == '__main__':
    # Specify paths and hyperparameters for quantization
    model_path = "/mnt/e/PyCharm/PreTrainModel/qwen_7b_chat_lora_merge"
    quant_path = "/mnt/e/PyCharm/PreTrainModel/qwen_7b_chat_lora_merge_gptq_4_test2"
    data_path = '/mnt/e/PyCharm/insteresting/LLaMA-Factory-0.7.1/data/qwen-7b-sql-gptq.json'

    # with open(data_path, 'r', encoding='utf-8') as f:
    #     messages = json.load(f)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # Load your tokenizer and model with AutoAWQ
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

    data = preprocess(data_path)

    model.quantize(tokenizer, quant_config=quant_config, calib_data=data)

    model.save_quantized(quant_path, safetensors=True)
    tokenizer.save_pretrained(quant_path)
