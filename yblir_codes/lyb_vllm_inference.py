# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 下午4:25
# @Author  : yblir
# @File    : lyb_vllm_inference.py
# explain  : 
# =======================================================
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    model_path='"Qwen/Qwen1.5-7B-Chat"'
    tokenizer = AutoTokenizer.from_pretrained()

    # Pass the default decoding hyperparameters of Qwen1.5-7B-Chat
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    llm = LLM(
            model="Qwen/Qwen1.5-7B-Chat",
            # tensor_parallel_size=4, # 模型并行分配到几张显卡上，不使用则加载到一张显卡上
            # quantization="awq", # 对于量化的模型，指定量化方法，有awq,gptq两个参数
            # kv_cache_dtype="fp8_e5m2" # kv缓存量化，可以与quantization一起使用
    )

    # Prepare your prompts
    prompt = "Tell me something about large language models."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")