# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午11:27
# @Author  : yblir
# @File    : lyb_autogptq2.py
# explain  : https://qwen.readthedocs.io/zh-cn/latest/quantization/gptq.html
# =======================================================
import logging
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def preprocess_text(messages):
    """
    最终处理后，msg格式如下，system要改成自己的：
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ]
    """
    data = []
    for msg in messages:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))
    return data


if __name__ == '__main__':
    # Specify paths and hyperparameters for quantization
    model_path = "your_model_path"
    quant_path = "your_quantized_model_path"

    quantize_config = BaseQuantizeConfig(
            bits=8,  # 4 or 8
            group_size=128,
            damp_percent=0.01,
            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
            static_groups=False,
            sym=True,
            true_sequential=True,
            model_name_or_path=None,
            model_file_base_name="model"
    )
    max_len = 8192

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config,
            device_map="auto",
            # 用多GPU来读取模型, 与device_map二选一
            # max_memory={i:"20GB" for i in range(4)},
            trust_remote_code=True
    )

    data = preprocess_text(["你好，我是机器人。"])

    model.quantize(data, cache_examples_on_gpu=False)

    model.save_quantized(quant_path, use_safetensors=True)
    tokenizer.save_pretrained(quant_path)
