# -*- coding: utf-8 -*-
# @Time    : 2024/5/25 10:15
# @Author  : yblir
# @File    : lyb_eval.py
# explain  : 
# =======================================================
import yaml
import json
from loguru import logger
import time
import sys
from llamafactory.eval.evaluator import Evaluator

if __name__ == '__main__':
    with open('../examples/yblir_configs/lyb_qwen_eval.yaml', 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)

    Evaluator(param).eval()
