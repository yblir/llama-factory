# -*- coding: utf-8 -*-
# @Time    : 2024/6/4 下午2:01
# @Author  : yblir
# @File    : lyb_qwen_vllm_infer.py
# explain  : 
# =======================================================
import json
import sys
import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

device = "cuda"  # the device to load the model onto


def qwen_preprocess(tokenizer_, msg):
    """
    最终处理后，msg格式如下，system要改成自己的：
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ]
    """

    # tokenizer.apply_chat_template() 与model.generate搭配使用
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": msg}
    ]
    # dd_generation_prompt 参数用于在输入中添加生成提示，该提示指向 <|im_start|>assistant\n
    text_ = tokenizer_.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # print(model_inputs)
    # sys.exit()
    return text_


if __name__ == '__main__':
    model_path = '/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b_merge_800'
    data_path = '/media/xk/D6B8A862B8A8433B/GitHub/llama-factory/data/train_clean_eval.json'

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 输出采样策略
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for i, item in enumerate(data):
        print(f'{i} ------------------------------------------')
        text = qwen_preprocess(tokenizer, item['instruction'])
        # generate outputs
        outputs = llm.generate([text], sampling_params)

        # Print the outputs.
        for output in outputs:
            # prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            print(generated_text)