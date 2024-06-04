# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午11:27
# @Author  : yblir
# @File    : lyb_autogptq2.py
# explain  : https://qwen.readthedocs.io/zh-cn/latest/quantization/gptq.html
# =======================================================
import logging
import json
import sys

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
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
        # print(input_ids)
        # sys.exit()
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer_.pad_token_id)))

    return data


if __name__ == '__main__':
    model_dir_path = "/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b_merge_800"
    quantized_path = "/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b_merge_800_int4_gptq"
    # 验证prompt的单条数据
    # quantize_dataset_path = '/media/xk/D6B8A862B8A8433B/GitHub/llama-factory/data/train_clean_test.json'
    quantize_dataset_path = '/media/xk/D6B8A862B8A8433B/GitHub/llama-factory/data/train_clean.json'

    # 加载校准集
    with open(quantize_dataset_path, 'r', encoding='utf-8') as f:
        lora_data = json.load(f)

    # 最大输入token数量，超出截断
    max_len = 8192
    quantize_config = BaseQuantizeConfig(
            # 有时fp16比量化后int4要快，这是因为原来有针对fp16的优化策略，在int4量化后无法使用，导致变慢
            bits=4,  # 4 or 8
            group_size=128,
            # 阻尼系数，用于量化过程中减少量化带来的震荡，例如，一个组中前一个量化损失小，后一个大，
            # 这参数大一点，那么前后两次量化损失差值就会小一点， 有什么效果呢？
            damp_percent=0.01,
            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
            # 是否使用静态组， 静态组简化计算，但精度下降
            static_groups=False,
            # 是否对称量化
            sym=True,
            # 是否使用真正的序列量化，True可以调高量化精度，但会增加计算量
            true_sequential=True,
            model_name_or_path=None,
            # 输出的权重，命名为model
            model_file_base_name="model"
    )
    # qwen1.5不再需要trust_remote_code=True,或许其他大模型需要吧
    tokenizer = AutoTokenizer.from_pretrained(model_dir_path, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
            model_dir_path,
            quantize_config,
            device_map="auto",
            # max_memory={i:"20GB" for i in range(4)}, # 用多GPU来读取模型, 与device_map二选一
            trust_remote_code=True
    )

    data = qwen_preprocess(lora_data, tokenizer, max_len)

    # cache_examples_on_gpu:中间量化缓存是否保存在gpu上,如果显存小,设为false. use_triton:使用triton加速包
    model.quantize(data, cache_examples_on_gpu=False, batch_size=1, use_triton=True)

    model.save_quantized(quantized_path, use_safetensors=True)
    tokenizer.save_pretrained(quantized_path)
