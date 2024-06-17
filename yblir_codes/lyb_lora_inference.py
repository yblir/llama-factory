# -*- coding: utf-8 -*-
# @Time    : 2024/5/16 23:50
# @Author  : yblir
# @File    : lyb_lora_inference.py
# explain  : 
# =======================================================
import yaml
from llamafactory.chat import ChatModel

if __name__ == '__main__':
    with open('../examples/yblir_configs/lyb_qwen_lora_merge_vllm.yaml', 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)

    chat_model = ChatModel(param)
    messages = []
    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            # torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        res=chat_model.chat(messages)
        print(res)
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
