# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(input_text, args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument(
            '--max_attention_window_size',
            type=int,
            default=None,
            help=
            'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    # parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='/home/TensorRT-LLM-0.9.0/checkpoints/qwen_7b_chat_int4_GPTQ/1-gpu')
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
            '--input_text',
            type=str,
            default=[input_text],
            nargs='+',
            # default=["Born in north-east France, Soyer trained as a"]
    )
    parser.add_argument(
            '--no_prompt_template',
            dest='use_prompt_template',
            default=True,
            action='store_false',
            help=
            "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=923)
    # parser.add_argument('--output_csv',
    #                     type=str,
    #                     help='CSV file where the tokenized output is stored.',
    #                     default=None)
    # parser.add_argument('--output_npy',
    #                     type=str,
    #                     help='Numpy file where the tokenized output is stored.',
    #                     default=None)
    # parser.add_argument(
    #     '--output_logits_npy',
    #     type=str,
    #     help=
    #     'Numpy file where the generation logits are stored. Use only when num_beams==1',
    #     default=None)

    # parser.add_argument('--output_log_probs_npy',
    #                     type=str,
    #                     help='Numpy file where the log_probs are stored',
    #                     default=None)
    #
    # parser.add_argument('--output_cum_log_probs_npy',
    #                     type=str,
    #                     help='Numpy file where the cum_log_probs are stored',
    #                     default=None)

    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='/home/TensorRT-LLM-0.9.0/checkpoints/qwen_7b_chat_lora_merge')
    parser.add_argument(
            '--tokenizer_type',
            help=
            'Specify that argument when providing a .model file as the tokenizer_dir. '
            'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams > 1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--early_stopping',
                        type=int,
                        help='Use early stopping if num_beams > 1'
                             '1 for early-stopping, 0 for non-early-stopping'
                             'other values for stopping by length',
                        default=1)
    # parser.add_argument('--debug_mode',
    #                     default=False,
    #                     action='store_true',
    #                     help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    # parser.add_argument(
    #     '--prompt_table_path',
    #     type=str,
    #     help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
            '--prompt_tasks',
            help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument(
            '--lora_task_uids',
            type=str,
            default=None,
            nargs="+",
            help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    # parser.add_argument(
    #     '--num_prepend_vtokens',
    #     nargs="+",
    #     type=int,
    #     help="Number of (default) virtual tokens to prepend to each sentence."
    #     " For example, '--num_prepend_vtokens=10' will prepend the tokens"
    #     " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    # parser.add_argument(
    #     '--run_profiling',
    #     default=False,
    #     action='store_true',
    #     help="Run several 10 iterations to profile the inference latencies.")
    # parser.add_argument(
    #     '--medusa_choices',
    #     type=str,
    #     default=None,
    #     help="Medusa choice to use, if not none, will use Medusa decoding."
    #     "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    # )

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                # num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    # if pad_id is None:
    #     pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    for curr_text in input_text:
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        input_ids = tokenizer.encode(curr_text,
                                     add_special_tokens=add_special_tokens,
                                     truncation=True,
                                     max_length=max_input_length)
        batch_input_ids.append(input_ids)

    # if num_prepend_vtokens:
    #     assert len(num_prepend_vtokens) == len(batch_input_ids)
    #     base_vocab_size = tokenizer.vocab_size - len(
    #             tokenizer.special_tokens_map.get('additional_special_tokens', []))
    #     for i, length in enumerate(num_prepend_vtokens):
    #         batch_input_ids[i] = list(
    #                 range(base_vocab_size,
    #                       base_vocab_size + length)) + batch_input_ids[i]

    # if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
    #     for ids in batch_input_ids:
    #         ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 # output_csv=None,
                 # output_npy=None,
                 # context_logits=None,
                 # generation_logits=None,
                 # cum_log_probs=None,
                 # log_probs=None,
                 # output_logits_npy=None,
                 # output_cum_log_probs_npy=None,
                 # output_log_probs_npy=None
                 ):
    batch_size, num_beams, _ = output_ids.size()
    for batch_idx in range(batch_size):
        inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist()
        input_text = tokenizer.decode(inputs)
        print(f'Input [Text {batch_idx}]: \"{input_text}\"')
        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][
                      output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            print(f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    # logger.set_level(args.log_level)

    model_name, model_version = read_model_name(args.engine_dir)

    tokenizer, pad_id, end_id = load_tokenizer(
            tokenizer_dir=args.tokenizer_dir,
            vocab_file=args.vocab_file,
            model_name=model_name,
            model_version=model_version,
            tokenizer_type=args.tokenizer_type,
    )

    stop_words_list = None
    bad_words_list = None
    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]
    print('input_text=', args.input_text)
    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  prompt_template=prompt_template,
                                  input_file=args.input_file,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id,
                                  # num_prepend_vtokens=args.num_prepend_vtokens,
                                  model_name=model_name,
                                  model_version=model_version)
    input_lengths = [x.size(0) for x in batch_input_ids]

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    # if args.debug_mode and not args.use_py_session:
    #     logger.warning(
    #             "Debug mode is not supported in C++ session for now, fallback to Python session."
    #     )
    #     args.use_py_session = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=False,
                         lora_ckpt_source=args.lora_ckpt_source)

    if not args.use_py_session:
        runner_kwargs.update(
                max_batch_size=len(batch_input_ids),
                max_input_len=max(input_lengths),
                max_output_len=args.max_output_len,
                max_beam_width=args.num_beams,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
        )
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=args.max_output_len,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                # output_cum_log_probs=(args.output_cum_log_probs_npy != None),
                # output_log_probs=(args.output_log_probs_npy != None),
                lora_uids=args.lora_task_uids,
                # prompt_table_path=args.prompt_table_path,
                prompt_tasks=args.prompt_tasks,
                # streaming=args.streaming,
                output_sequence_lengths=True,
                return_dict=True,
                # medusa_choices=args.medusa_choices
        )
        torch.cuda.synchronize()

    if runtime_rank == 0:
        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        # context_logits = None
        # generation_logits = None
        # cum_log_probs = None
        # log_probs = None
        # if runner.gather_context_logits:
        #     context_logits = outputs['context_logits']
        # if runner.gather_generation_logits:
        #     generation_logits = outputs['generation_logits']
        # if args.output_cum_log_probs_npy != None:
        #     cum_log_probs = outputs['cum_log_probs']
        # if args.output_log_probs_npy != None:
        #     log_probs = outputs['log_probs']
        print_output(tokenizer,
                     output_ids,
                     input_lengths,
                     sequence_lengths,
                     # output_csv=args.output_csv,
                     # output_npy=args.output_npy,
                     # context_logits=context_logits,
                     # generation_logits=generation_logits,
                     # output_logits_npy=args.output_logits_npy,
                     # cum_log_probs=cum_log_probs,
                     # log_probs=log_probs,
                     # output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                     # output_log_probs_npy=args.output_log_probs_npy
                     )


if __name__ == '__main__':
    input_text = '你是谁呀??'
    args = parse_arguments(input_text)
    # args.input_text='你是谁呀??'
    main(args)
    input_text = '介绍下你自己'
    print('=========================================================')
    args = parse_arguments(input_text)
    # args.input_text='你是谁呀??'
    main(args)
