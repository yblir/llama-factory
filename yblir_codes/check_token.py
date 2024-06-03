# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 上午11:05
# @Author  : yblir
# @File    : check_token.py
# explain  : 
# =======================================================
import torch
from transformers import AutoTokenizer

message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role"   : "user",
     "content": "<question>:查询所有考试信息. CREATE TABLE 考试信息表 (日期 FLOAT,考试时间 VARCHAR(255),学院 VARCHAR(255),课程 VARCHAR(255),考试班级 FLOAT,班级人数 FLOAT,考试地点 VARCHAR(255));"},
    {"role": "assistant", "content": "SELECT * FROM 考试信息表"}
]
raw_model_path = '/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b'
max_len = 8192
tokenizer = AutoTokenizer.from_pretrained(raw_model_path)
print(tokenizer.chat_template)
print('--------------------------------------------')
text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
text=text.strip()
print('text=',text)
b=tokenizer.tokenize(text)
print('b=',b)
c=tokenizer.convert_tokens_to_ids(b)
print('c=',c)
model_inputs = tokenizer([text])
print('model_inputs:',model_inputs)
input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)

# print('input_ids=', input_ids)
# print('attention_mask=', input_ids.ne(tokenizer.pad_token_id))
a=tokenizer.convert_ids_to_tokens([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,     27,   7841,  26818,  51154,
          55338, 103960,  27369,     13,  30776,  14363,   8908,    222,    225,
          41321,  27369,  20742,    320,  45785,  50116,     11, 103960,  20450,
          37589,      7,     17,     20,     20,    701, 101085,  37589,      7,
             17,     20,     20,    701, 103995,  37589,      7,     17,     20,
             20,    701, 103960, 107278,  50116,     11, 107278, 104346,  50116,
             11, 103960, 104766,  37589,      7,     17,     20,     20,   5905,
         151645,    198, 151644,  77091,    198,   4858,    353,   4295,   8908,
            222,    225,  41321,  27369,  20742, 151645,198])

# a=tokenizer.convert_ids_to_tokens([151644,   8948,    198])
# print(a)