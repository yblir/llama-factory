# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午11:27
# @Author  : yblir
# @File    : lyb_autogptq.py
# explain  : 
# =======================================================
from transformers import AutoModelForCausalLM,AutoTokenizer,GPTQConfig,AwqConfig


model_id = "facebook/opt-125m"
save_path = "path/to/save/model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=gptq_config
)

model.save_pretrained(save_path)