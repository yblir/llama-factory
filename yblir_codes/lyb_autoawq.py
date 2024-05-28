# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 上午10:55
# @Author  : yblir
# @File    : lyb_autoawq.py
# explain  : 
# =======================================================
# https://huggingface.co/docs/transformers/main/zh/main_classes/quantization
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "TheBloke/zephyr-7B-alpha-AWQ"

model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        # 将AWQ量化与Flash Attention结合起来，得到一个既被量化又更快速的模型
        attn_implementation="flash_attention_2",
        device_map="cuda:0"
)
