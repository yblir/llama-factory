# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午10:55
# @Author  : yblir
# @File    : lyb_autoawq_qwen.py
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
        model_inputs = tokenizer_([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len_], dtype=torch.int)
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer_.pad_token_id)))

    return data

# todo 不能运行
if __name__ == '__main__':
    # Specify paths and hyperparameters for quantization
    model_path = "/mnt/e/PyCharm/PreTrainModel/qwen_7b_chat_lora_merge"
    quant_path = "/mnt/e/PyCharm/PreTrainModel/qwen_7b_chat_lora_merge_awq_4_test2"
    data_path = '/mnt/e/PyCharm/insteresting/LLaMA-Factory-0.7.1/data/qwen-7b-sql-gptq.json'

    with open(data_path, 'r', encoding='utf-8') as f:
        lora_data = json.load(f)

    max_len=2048
    # GEMM:文本长或batch_size比较大时，速度会快，少文本时，GEMV会更快
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # Load your tokenizer and model with AutoAWQ
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

    data = qwen_preprocess(lora_data, tokenizer, max_len)

    # 默认数据长度大于512，该条数据不使用
    model.quantize(tokenizer, quant_config=quant_config, calib_data=data)

    model.save_quantized(quant_path, safetensors=True)
    tokenizer.save_pretrained(quant_path)
