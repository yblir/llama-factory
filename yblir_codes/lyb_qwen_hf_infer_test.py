# -*- coding: utf-8 -*-
# @Time    : 2024/6/4 下午12:14
# @Author  : yblir
# @File    : lyb_qwen_hf_infer.py
# explain  : 
# =======================================================
import json
import sys
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer,GPT2Tokenizer,GPT2Model,set_seed

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
    # msg='说吧，你到底是谁'
    # tokenizer.apply_chat_template() 与model.generate搭配使用
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": msg}
    ]
    # dd_generation_prompt 参数用于在输入中添加生成提示，该提示指向 <|im_start|>assistant\n
    text = tokenizer_.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs_ = tokenizer_([text], return_tensors="pt").to(device)

    input_ids = tokenizer_.encode(text, return_tensors='pt')
    attention_mask_ = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    # print(model_inputs)
    # sys.exit()
    return model_inputs_, attention_mask_


if __name__ == '__main__':

    # model_path = '/media/xk/D6B8A862B8A8433B/data/gpt2'
    # model_path = '/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b'
    model_path = '/media/xk/D6B8A862B8A8433B/data/qwen1_5-1_8b_merge_800'
    data_path = '/media/xk/D6B8A862B8A8433B/GitHub/llama-factory/data/train_clean_test.json'

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2"
            pad_token_id=tokenizer.eos_token_id
    )

    for i, item in enumerate(data):
        print(f'{i} ------------------------------------------')
        # item['instruction'] = '你是谁，介绍下自己'
        model_inputs, attention_mask = qwen_preprocess(tokenizer, item['instruction'])
        prompt_length = len(model_inputs.input_ids[0])

        generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=50,  # 最大输出长度.
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # 是否使用采样；False:使用贪婪解码
                num_beams=3,

                # 联合采样 顺序是 top-k -> top-p -> Temperature
                # 表示保留概率最高的top-k个 token,保留概率的累计和达到 top-p 的单词,接下来使用Temperature = 0.x 进行归一化
                top_k=50,
                top_p=0.95,  # top-P越高，候选词越多，多样性越丰富。top-P越低，候选词越少，越稳定

                # 一般来说，prompt 越长，描述得越清楚，模型生成的输出质量就越好，置信度越高，这时可以适当调高 temperature 的值；
                # 反过来，如果 prompt 很短，很含糊，这时再设置一个比较高的 temperature 值，模型的输出就很不稳定了。
                temperature=0.8,  # 控制生成文本多样性的参数，温度越高，生成的文本越随机。温度越低，概率分布差距越大，越容易采样到概率大的字

                # repetition_penalty=2.0,

                # early_stopping=True,

                # max_length=50,
                # num_beams=5,
                # no_repeat_ngram_size=2,
                # num_return_sequences=5,
                # early_stopping=True

                # do_sample=True,
                # max_length=50,
                # top_k=0,
                # temperature=0.7,

                # do_sample=True,
                # max_length=50,
                # top_k=50

                # do_sample=True,
                # max_length=50,
                # top_p=0.92,
                # top_k=0,

                num_return_sequences=3
        )
        # print(generated_ids)
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        generated_ids = generated_ids.tolist()

        output_ids = []
        for i in range(len(generated_ids)):
            output_ids.append(generated_ids[i][prompt_length:])
        # print(generated_ids)

        responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for response in responses:
            print(response)

        # for output in output_ids:
        #     response = tokenizer.batch_decode(output, skip_special_tokens=True)
        #     print(' '.join(response))

        # print(response)
