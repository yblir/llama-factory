# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午11:27
# @Author  : yblir
# @File    : lyb_autogptq2.py
# explain  : https://qwen.readthedocs.io/zh-cn/latest/quantization/gptq.html
# =======================================================
import logging
import json
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def qwen_preprocess(data_path_):
    """
    最终处理后，msg格式如下，system要改成自己的：
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ]
    """
    with open(data_path_, 'r', encoding='utf-8') as f:
        lora_data = json.load(f)

    messages = []
    for item in lora_data:
        temp = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item['instruction']},
            {"role": "assistant", "content": item['output']}
        ]
        messages.append(temp)

    data = []
    for msg in messages:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))

    return data


if __name__ == '__main__':
    # Specify paths and hyperparameters for quantization
    model_path = "/mnt/e/PyCharm/PreTrainModel/qwen_7b_chat_lora_merge"
    quant_path = "/mnt/e/PyCharm/PreTrainModel/qwen_7b_chat_lora_merge_gptq_4_test2"
    quantize_dataset_path = '/mnt/e/PyCharm/insteresting/LLaMA-Factory-0.7.1/data/qwen-7b-sql-gptq.json'

    # 最大输入token数量，超出截断
    max_len = 8192
    quantize_config = BaseQuantizeConfig(
            bits=4,  # 4 or 8
            group_size=128,
            damp_percent=0.01,
            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
            static_groups=False,
            sym=True,
            true_sequential=True,
            model_name_or_path=None,
            model_file_base_name="model"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config,
            device_map="auto",
            # max_memory={i:"20GB" for i in range(4)}, # 用多GPU来读取模型, 与device_map二选一
            trust_remote_code=True
    )

    data = qwen_preprocess(quantize_dataset_path)

    # cache_examples_on_gpu:中间量化缓存是否保存在gpu上,如果显存小,设为false. use_triton:使用triton加速包
    model.quantize(data, cache_examples_on_gpu=False,batch_size=1,use_triton=True)

    model.save_quantized(quant_path, use_safetensors=True)
    tokenizer.save_pretrained(quant_path)
