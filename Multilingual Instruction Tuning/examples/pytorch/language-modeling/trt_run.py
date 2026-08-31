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
import csv
import json
from pathlib import Path

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument("--method", default='plug')
    parser.add_argument('--alpaca_lang', default='')
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
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument("--output_file", default='')
    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='gpt2')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
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
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument("--split_num", default=0, type=int)
    parser.add_argument("--total_splits", default=8, type=int)
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")

    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.read()
                input_ids = tokenizer.encode(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)
                batch_input_ids.append(input_ids)
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]
    if model_name == 'glm_10b':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 output_logits_npy=None):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            write_dict = {}
            write_dict['inputs'] = input_text
            # print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                write_dict[f"outputs{beam}"] = output_text
                # print(
                #     f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')
            output_file.write(json.dumps(write_dict) + '\n')

    # output_ids = output_ids.reshape((-1, output_ids.size(2)))
    #
    # if output_csv is not None:
    #     output_file = Path(output_csv)
    #     output_file.parent.mkdir(exist_ok=True, parents=True)
    #     outputs = output_ids.tolist()
    #     with open(output_file, 'w') as csv_file:
    #         writer = csv.writer(csv_file, delimiter=',')
    #         writer.writerows(outputs)
    #
    # if output_npy is not None:
    #     output_file = Path(output_npy)
    #     output_file.parent.mkdir(exist_ok=True, parents=True)
    #     outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
    #     np.save(output_file, outputs)
    #
    # if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
    #     input_lengths = torch.Tensor(input_lengths)
    #     context_logits = torch.cat(context_logits, axis=0)
    #     generation_logits = generation_logits.squeeze(1)
    #     last_token_ids = torch.cumsum(input_lengths, dim=0).int().cuda()
    #     batch_size = input_lengths.size(0)
    #     vocab_size_padded = context_logits.shape[-1]
    #     context_logits = context_logits.reshape([1, -1, vocab_size_padded])
    #     contex_last_token_logits = torch.index_select(context_logits, 1,
    #                                                   last_token_ids - 1).view(
    #                                                       batch_size, 1,
    #                                                       vocab_size_padded)
    #     generation_logits = torch.cat(
    #         [contex_last_token_logits, generation_logits], axis=1)
    #     generation_logits = generation_logits.reshape(
    #         -1, num_beams, generation_logits.shape[1], generation_logits.
    #         shape[2])  # [batchSize, beamWidth, maxOutputLength, vocabSize]
    #     # Save context logits
    #     output_context_logits_npy = output_logits_npy.split(
    #         '.npy')[0] + "_context"
    #     output_context_logits_file = Path(output_context_logits_npy)
    #     context_outputs = np.array(
    #         context_logits.squeeze(0).cpu().contiguous(),
    #         dtype='float32')  # [promptLengthSum, vocabSize]
    #     np.save(output_context_logits_file, context_outputs)
    #     # Save generation logits
    #     output_generation_logits_npy = output_logits_npy.split(
    #         '.npy')[0] + "_generation"
    #     output_generation_logits_file = Path(output_generation_logits_npy)
    #     generation_outputs = np.array(generation_logits.cpu().contiguous(),
    #                                   dtype='float32')
    #     np.save(output_generation_logits_file, generation_outputs)


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    # runtime_rank = 0
    logger.set_level(args.log_level)

    model_name = read_model_name(args.engine_dir)
    if args.tokenizer_dir is None:
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        tokenizer_type=args.tokenizer_type,
    )

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    def tokenize_alpaca(example, output_prompt_only=False):
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
        lang_to_prompt_dict = {
            'en': PROMPT_DICT,
            "ht": {
                "prompt_input": "Anba a se yon enstriksyon ki dekri yon travay, ansanm ak yon opinyon ki bay plis kont\u00e8ks. Ekri yon repons ki byen ranpli demann lan.\n\n### Enstriksyon:\n{instruction}\n\n### Antre:\n{input}\n\n### Repons:",
                "prompt_no_input": "Anba a se yon enstriksyon ki dekri yon travay. Ekri yon repons ki byen ranpli demann lan.\n\n### Enstriksyon:\n{instruction}\n\n### Repons:"
            },
            "qu": {
                "prompt_input": "Uraypiqa huk kamachiymi kachkan, chaymi huk ruwayta willan, huk yaykusqawan tupachisqa, chaymi aswan contextota qun. Ma\u00f1akusqata allinta hunt\u2019aq kutichiyta qillqay.\n\n### Yachachiy:\n{instruction}.\n\n### Yaykuchiy:\n{input}.\n\n### Kutichiy:",
                "prompt_no_input": "Uraypiqa huk kamachikuymi kachkan, chaypim huk llamkaymanta willakun. Ma\u00f1akusqata allinta hunt\u2019aq kutichiyta qillqay.\n\n### Yachachiy:\n{instruction}.\n\n### Kutichiy:"
            },
            "te": {
                "prompt_input": "\u0c26\u0c3f\u0c17\u0c41\u0c35\u0c28 \u0c12\u0c15 \u0c2a\u0c28\u0c3f\u0c28\u0c3f \u0c35\u0c3f\u0c35\u0c30\u0c3f\u0c02\u0c1a\u0c47 \u0c38\u0c42\u0c1a\u0c28, \u0c24\u0c26\u0c41\u0c2a\u0c30\u0c3f \u0c38\u0c02\u0c26\u0c30\u0c4d\u0c2d\u0c3e\u0c28\u0c4d\u0c28\u0c3f \u0c05\u0c02\u0c26\u0c3f\u0c02\u0c1a\u0c47 \u0c07\u0c28\u0c4d\u200c\u0c2a\u0c41\u0c1f\u0c4d\u200c\u0c24\u0c4b \u0c1c\u0c24 \u0c1a\u0c47\u0c2f\u0c2c\u0c21\u0c3f\u0c02\u0c26\u0c3f. \u0c05\u0c2d\u0c4d\u0c2f\u0c30\u0c4d\u0c25\u0c28\u0c28\u0c41 \u0c38\u0c2e\u0c41\u0c1a\u0c3f\u0c24\u0c02\u0c17\u0c3e \u0c2a\u0c42\u0c30\u0c4d\u0c24\u0c3f \u0c1a\u0c47\u0c38\u0c47 \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28\u0c28\u0c41 \u0c35\u0c4d\u0c30\u0c3e\u0c2f\u0c02\u0c21\u0c3f.\n\n### \u0c38\u0c42\u0c1a\u0c28:\n{instruction}\n\n### \u0c07\u0c28\u0c4d\u200c\u0c2a\u0c41\u0c1f\u0c4d:\n{input}\n\n### \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28:",
                "prompt_no_input": "\u0c15\u0c4d\u0c30\u0c3f\u0c02\u0c26 \u0c12\u0c15 \u0c2a\u0c28\u0c3f\u0c28\u0c3f \u0c35\u0c3f\u0c35\u0c30\u0c3f\u0c02\u0c1a\u0c47 \u0c38\u0c42\u0c1a\u0c28 \u0c09\u0c02\u0c26\u0c3f. \u0c05\u0c2d\u0c4d\u0c2f\u0c30\u0c4d\u0c25\u0c28\u0c28\u0c41 \u0c38\u0c2e\u0c41\u0c1a\u0c3f\u0c24\u0c02\u0c17\u0c3e \u0c2a\u0c42\u0c30\u0c4d\u0c24\u0c3f \u0c1a\u0c47\u0c38\u0c47 \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28\u0c28\u0c41 \u0c35\u0c4d\u0c30\u0c3e\u0c2f\u0c02\u0c21\u0c3f.\n\n### \u0c38\u0c42\u0c1a\u0c28:\n{instruction}\n\n### \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28:"
            },
            "sw": {
                "prompt_input": "Ifuatayo ni maagizo ambayo yanaelezea kazi, yakioanishwa na ingizo ambalo hutoa muktadha zaidi. Andika jibu ambalo linakamilisha ombi ipasavyo.\n\n### Maagizo:\n{instruction}\n\n### Ingizo:\n{input}\n\n### Jibu:",
                "prompt_no_input": "Chini ni maagizo ambayo yanaelezea kazi. Andika jibu ambalo linakamilisha ombi ipasavyo.\n\n### Maagizo:\n{instruction}\n\n### Jibu:"
            },
            "ta": {
                "prompt_input": "\u0b92\u0bb0\u0bc1 \u0baa\u0ba3\u0bbf\u0baf\u0bc8 \u0bb5\u0bbf\u0bb5\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc1\u0bae\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd \u0b95\u0bc0\u0bb4\u0bc7 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1, \u0bae\u0bc7\u0bb2\u0bc1\u0bae\u0bcd \u0b9a\u0bc2\u0bb4\u0bb2\u0bc8 \u0bb5\u0bb4\u0b99\u0bcd\u0b95\u0bc1\u0bae\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0bc0\u0b9f\u0bcd\u0b9f\u0bc1\u0b9f\u0ba9\u0bcd \u0b87\u0ba3\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b95\u0bcb\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc8\u0baf\u0bc8 \u0b9a\u0bb0\u0bbf\u0baf\u0bbe\u0ba9 \u0bae\u0bc1\u0bb1\u0bc8\u0baf\u0bbf\u0bb2\u0bcd \u0ba8\u0bbf\u0bb1\u0bc8\u0bb5\u0bc1 \u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0bc1\u0bae\u0bcd \u0baa\u0ba4\u0bbf\u0bb2\u0bc8 \u0b8e\u0bb4\u0bc1\u0ba4\u0bb5\u0bc1\u0bae\u0bcd.\n\n### \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd:\n{instruction}\n\n### \u0b89\u0bb3\u0bcd\u0bb3\u0bc0\u0b9f\u0bc1:\n{input}\n\n### \u0baa\u0ba4\u0bbf\u0bb2\u0bcd:",
                "prompt_no_input": "\u0b92\u0bb0\u0bc1 \u0baa\u0ba3\u0bbf\u0baf\u0bc8 \u0bb5\u0bbf\u0bb5\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc1\u0bae\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd \u0b95\u0bc0\u0bb4\u0bc7 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b95\u0bcb\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc8\u0baf\u0bc8 \u0b9a\u0bb0\u0bbf\u0baf\u0bbe\u0ba9 \u0bae\u0bc1\u0bb1\u0bc8\u0baf\u0bbf\u0bb2\u0bcd \u0ba8\u0bbf\u0bb1\u0bc8\u0bb5\u0bc1 \u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0bc1\u0bae\u0bcd \u0baa\u0ba4\u0bbf\u0bb2\u0bc8 \u0b8e\u0bb4\u0bc1\u0ba4\u0bb5\u0bc1\u0bae\u0bcd.\n\n### \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd:\n{instruction}\n\n### \u0baa\u0ba4\u0bbf\u0bb2\u0bcd:"
            },
            "ur": {
                "prompt_input": "\u0630\u06cc\u0644 \u0645\u06cc\u06ba \u0627\u06cc\u06a9 \u06c1\u062f\u0627\u06cc\u062a \u062f\u06cc \u06af\u0626\u06cc \u06c1\u06d2 \u062c\u0648 \u0627\u06cc\u06a9 \u06a9\u0627\u0645 \u06a9\u06cc \u0648\u0636\u0627\u062d\u062a \u06a9\u0631\u062a\u06cc \u06c1\u06d2\u060c \u0627\u06cc\u06a9 \u0627\u0646 \u067e\u0679 \u06a9\u06d2 \u0633\u0627\u062a\u06be \u062c\u0648\u0691\u0627 \u062c\u0648 \u0645\u0632\u06cc\u062f \u0633\u06cc\u0627\u0642 \u0648 \u0633\u0628\u0627\u0642 \u0641\u0631\u0627\u06c1\u0645 \u06a9\u0631\u062a\u0627 \u06c1\u06d2\u06d4 \u0627\u06cc\u06a9 \u062c\u0648\u0627\u0628 \u0644\u06a9\u06be\u06cc\u06ba \u062c\u0648 \u0645\u0646\u0627\u0633\u0628 \u0637\u0631\u06cc\u0642\u06d2 \u0633\u06d2 \u062f\u0631\u062e\u0648\u0627\u0633\u062a \u06a9\u0648 \u0645\u06a9\u0645\u0644 \u06a9\u0631\u06d2\u06d4\n\n### \u06c1\u062f\u0627\u06cc\u0627\u062a:\n{instruction}\n\n### \u0627\u0646 \u067e\u0679:\n{input}\n\n### \u062c\u0648\u0627\u0628:",
                "prompt_no_input": "\u0630\u06cc\u0644 \u0645\u06cc\u06ba \u0627\u06cc\u06a9 \u06c1\u062f\u0627\u06cc\u062a \u06c1\u06d2 \u062c\u0648 \u0627\u06cc\u06a9 \u06a9\u0627\u0645 \u06a9\u06cc \u0648\u0636\u0627\u062d\u062a \u06a9\u0631\u062a\u06cc \u06c1\u06d2\u06d4 \u0627\u06cc\u06a9 \u062c\u0648\u0627\u0628 \u0644\u06a9\u06be\u06cc\u06ba \u062c\u0648 \u0645\u0646\u0627\u0633\u0628 \u0637\u0631\u06cc\u0642\u06d2 \u0633\u06d2 \u062f\u0631\u062e\u0648\u0627\u0633\u062a \u06a9\u0648 \u0645\u06a9\u0645\u0644 \u06a9\u0631\u06d2\u06d4\n\n### \u06c1\u062f\u0627\u06cc\u0627\u062a:\n{instruction}\n\n### \u062c\u0648\u0627\u0628:"
            },
            "my": {
                "prompt_input": "\u1021\u1031\u102c\u1000\u103a\u1010\u103d\u1004\u103a \u1014\u1031\u102c\u1000\u103a\u1011\u1015\u103a\u1021\u1000\u103c\u1031\u102c\u1004\u103a\u1038\u1021\u101b\u102c\u1010\u1005\u103a\u1001\u102f\u1000\u102d\u102f \u1015\u1036\u1037\u1015\u102d\u102f\u1038\u1015\u1031\u1038\u101e\u100a\u1037\u103a \u1011\u100a\u1037\u103a\u101e\u103d\u1004\u103a\u1038\u1019\u103e\u102f\u1010\u1005\u103a\u1001\u102f\u1014\u103e\u1004\u1037\u103a \u1010\u103d\u1032\u101c\u102f\u1015\u103a\u1011\u102c\u1038\u101e\u100a\u1037\u103a \u1021\u101c\u102f\u1015\u103a\u1010\u1005\u103a\u1001\u102f\u1000\u102d\u102f \u1016\u1031\u102c\u103a\u1015\u103c\u101e\u100a\u1037\u103a \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a\u1010\u1005\u103a\u1001\u102f\u1016\u103c\u1005\u103a\u101e\u100a\u103a\u104b \u1010\u1031\u102c\u1004\u103a\u1038\u1006\u102d\u102f\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101e\u1004\u1037\u103a\u101c\u103b\u1031\u102c\u103a\u1005\u103d\u102c \u1015\u103c\u102e\u1038\u1019\u103c\u1031\u102c\u1000\u103a\u1005\u1031\u101e\u1031\u102c \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101b\u1031\u1038\u1015\u102b\u104b\n\n### \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a-\n{instruction}\n\n### \u1011\u100a\u1037\u103a\u101e\u103d\u1004\u103a\u1038\u1019\u103e\u102f-\n{input}\n\n### \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1019\u103e\u102f-",
                "prompt_no_input": "\u1021\u1031\u102c\u1000\u103a\u1010\u103d\u1004\u103a \u1021\u101c\u102f\u1015\u103a\u1010\u1005\u103a\u1001\u102f\u1000\u102d\u102f \u1016\u1031\u102c\u103a\u1015\u103c\u101e\u100a\u1037\u103a \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a\u1010\u1005\u103a\u1001\u102f\u1016\u103c\u1005\u103a\u101e\u100a\u103a\u104b \u1010\u1031\u102c\u1004\u103a\u1038\u1006\u102d\u102f\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101e\u1004\u1037\u103a\u101c\u103b\u1031\u102c\u103a\u1005\u103d\u102c \u1015\u103c\u102e\u1038\u1019\u103c\u1031\u102c\u1000\u103a\u1005\u1031\u101e\u1031\u102c \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101b\u1031\u1038\u1015\u102b\u104b\n\n### \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a-\n{instruction}\n\n### \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1019\u103e\u102f-"
            },
            "hi": {
                "prompt_input": "\u0928\u0940\u091a\u0947 \u090f\u0915 \u0928\u093f\u0930\u094d\u0926\u0947\u0936 \u0939\u0948 \u091c\u094b \u0915\u093f\u0938\u0940 \u0915\u093e\u0930\u094d\u092f \u0915\u093e \u0935\u0930\u094d\u0923\u0928 \u0915\u0930\u0924\u093e \u0939\u0948, \u091c\u093f\u0938\u0947 \u090f\u0915 \u0907\u0928\u092a\u0941\u091f \u0915\u0947 \u0938\u093e\u0925 \u091c\u094b\u0921\u093c\u093e \u0917\u092f\u093e \u0939\u0948 \u091c\u094b \u0906\u0917\u0947 \u0915\u093e \u0938\u0902\u0926\u0930\u094d\u092d \u092a\u094d\u0930\u0926\u093e\u0928 \u0915\u0930\u0924\u093e \u0939\u0948\u0964 \u0910\u0938\u093e \u0909\u0924\u094d\u0924\u0930 \u0932\u093f\u0916\u0947\u0902 \u091c\u094b \u0905\u0928\u0941\u0930\u094b\u0927 \u0915\u094b \u0909\u091a\u093f\u0924 \u0930\u0942\u092a \u0938\u0947 \u092a\u0942\u0930\u093e \u0915\u0930\u0924\u093e \u0939\u094b\u0964\n\n### \u0928\u093f\u0930\u094d\u0926\u0947\u0936:\n{instruction}\n\n### \u0907\u0928\u092a\u0941\u091f:\n{input}\n\n### \u092a\u094d\u0930\u0924\u093f\u0915\u094d\u0930\u093f\u092f\u093e:",
                "prompt_no_input": "\u0928\u0940\u091a\u0947 \u090f\u0915 \u0928\u093f\u0930\u094d\u0926\u0947\u0936 \u0939\u0948 \u091c\u094b \u0915\u093f\u0938\u0940 \u0915\u093e\u0930\u094d\u092f \u0915\u093e \u0935\u0930\u094d\u0923\u0928 \u0915\u0930\u0924\u093e \u0939\u0948\u0964 \u0910\u0938\u093e \u0909\u0924\u094d\u0924\u0930 \u0932\u093f\u0916\u0947\u0902 \u091c\u094b \u0905\u0928\u0941\u0930\u094b\u0927 \u0915\u094b \u0909\u091a\u093f\u0924 \u0930\u0942\u092a \u0938\u0947 \u092a\u0942\u0930\u093e \u0915\u0930\u0924\u093e \u0939\u094b\u0964\n\n### \u0928\u093f\u0930\u094d\u0926\u0947\u0936:\n{instruction}\n\n### \u092a\u094d\u0930\u0924\u093f\u0915\u094d\u0930\u093f\u092f\u093e:"
            },
            "th": {
                "prompt_input": "\u0e14\u0e49\u0e32\u0e19\u0e25\u0e48\u0e32\u0e07\u0e19\u0e35\u0e49\u0e04\u0e37\u0e2d\u0e04\u0e33\u0e2a\u0e31\u0e48\u0e07\u0e17\u0e35\u0e48\u0e2d\u0e18\u0e34\u0e1a\u0e32\u0e22\u0e07\u0e32\u0e19 \u0e04\u0e27\u0e1a\u0e04\u0e39\u0e48\u0e44\u0e1b\u0e01\u0e31\u0e1a\u0e2d\u0e34\u0e19\u0e1e\u0e38\u0e15\u0e17\u0e35\u0e48\u0e43\u0e2b\u0e49\u0e1a\u0e23\u0e34\u0e1a\u0e17\u0e40\u0e1e\u0e34\u0e48\u0e21\u0e40\u0e15\u0e34\u0e21 \u0e40\u0e02\u0e35\u0e22\u0e19\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e48\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07\u0e04\u0e33\u0e02\u0e2d\u0e44\u0e14\u0e49\u0e2d\u0e22\u0e48\u0e32\u0e07\u0e40\u0e2b\u0e21\u0e32\u0e30\u0e2a\u0e21\n\n### \u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33:\n{instruction}\n\n### \u0e1b\u0e49\u0e2d\u0e19\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25:\n{input}\n\n### \u0e01\u0e32\u0e23\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07:",
                "prompt_no_input": "\u0e14\u0e49\u0e32\u0e19\u0e25\u0e48\u0e32\u0e07\u0e19\u0e35\u0e49\u0e40\u0e1b\u0e47\u0e19\u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33\u0e17\u0e35\u0e48\u0e2d\u0e18\u0e34\u0e1a\u0e32\u0e22\u0e07\u0e32\u0e19 \u0e40\u0e02\u0e35\u0e22\u0e19\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e48\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07\u0e04\u0e33\u0e02\u0e2d\u0e44\u0e14\u0e49\u0e2d\u0e22\u0e48\u0e32\u0e07\u0e40\u0e2b\u0e21\u0e32\u0e30\u0e2a\u0e21\n\n### \u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33:\n{instruction}\n\n### \u0e01\u0e32\u0e23\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07:"
            },
            "ar": {
                "prompt_input": "\u064a\u0648\u062c\u062f \u0623\u062f\u0646\u0627\u0647 \u062a\u0639\u0644\u064a\u0645\u0627\u062a \u062a\u0635\u0641 \u0645\u0647\u0645\u0629\u060c \u0645\u0642\u062a\u0631\u0646\u0629 \u0628\u0625\u062f\u062e\u0627\u0644 \u064a\u0648\u0641\u0631 \u0633\u064a\u0627\u0642\u064b\u0627 \u0625\u0636\u0627\u0641\u064a\u064b\u0627. \u0627\u0643\u062a\u0628 \u0627\u0644\u0631\u062f \u0627\u0644\u0630\u064a \u064a\u0643\u0645\u0644 \u0627\u0644\u0637\u0644\u0628 \u0628\u0634\u0643\u0644 \u0645\u0646\u0627\u0633\u0628.\n\n### \u062a\u0639\u0644\u064a\u0645\u0627\u062a:\n{instruction}\n\n### \u0645\u062f\u062e\u0644:\n{input}\n\n### \u0625\u062c\u0627\u0628\u0629:",
                "prompt_no_input": "\u064a\u0648\u062c\u062f \u0623\u062f\u0646\u0627\u0647 \u062a\u0639\u0644\u064a\u0645\u0627\u062a \u062a\u0635\u0641 \u0627\u0644\u0645\u0647\u0645\u0629. \u0627\u0643\u062a\u0628 \u0627\u0644\u0631\u062f \u0627\u0644\u0630\u064a \u064a\u0643\u0645\u0644 \u0627\u0644\u0637\u0644\u0628 \u0628\u0634\u0643\u0644 \u0645\u0646\u0627\u0633\u0628.\n\n### \u062a\u0639\u0644\u064a\u0645\u0627\u062a:\n{instruction}\n\n### \u0625\u062c\u0627\u0628\u0629:"
            },
            "bn": {
                "prompt_input": "\u09a8\u09c0\u099a\u09c7 \u098f\u0995\u099f\u09bf \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6 \u09b0\u09af\u09bc\u09c7\u099b\u09c7 \u09af\u09be \u098f\u0995\u099f\u09bf \u099f\u09be\u09b8\u09cd\u0995 \u09ac\u09b0\u09cd\u09a3\u09a8\u09be \u0995\u09b0\u09c7, \u098f\u0995\u099f\u09bf \u0987\u09a8\u09aa\u09c1\u099f\u09c7\u09b0 \u09b8\u09be\u09a5\u09c7 \u09af\u09c1\u0995\u09cd\u09a4 \u09af\u09be \u0986\u09b0\u0993 \u09aa\u09cd\u09b0\u09b8\u0999\u09cd\u0997 \u09b8\u09b0\u09ac\u09b0\u09be\u09b9 \u0995\u09b0\u09c7\u0964 \u098f\u0995\u099f\u09bf \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be \u09b2\u09bf\u0996\u09c1\u09a8 \u09af\u09be \u09af\u09a5\u09be\u09af\u09a5\u09ad\u09be\u09ac\u09c7 \u0985\u09a8\u09c1\u09b0\u09cb\u09a7\u099f\u09bf \u09b8\u09ae\u09cd\u09aa\u09c2\u09b0\u09cd\u09a3 \u0995\u09b0\u09c7\u0964\n\n### \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6:\n{instruction}\n\n### \u0987\u09a8\u09aa\u09c1\u099f:\n{input}\n\n### \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be:",
                "prompt_no_input": "\u09a8\u09c0\u099a\u09c7 \u098f\u0995\u099f\u09bf \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6 \u09af\u09be \u098f\u0995\u099f\u09bf \u099f\u09be\u09b8\u09cd\u0995 \u09ac\u09b0\u09cd\u09a3\u09a8\u09be \u0995\u09b0\u09c7\u0964 \u098f\u0995\u099f\u09bf \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be \u09b2\u09bf\u0996\u09c1\u09a8 \u09af\u09be \u09af\u09a5\u09be\u09af\u09a5\u09ad\u09be\u09ac\u09c7 \u0985\u09a8\u09c1\u09b0\u09cb\u09a7\u099f\u09bf \u09b8\u09ae\u09cd\u09aa\u09c2\u09b0\u09cd\u09a3 \u0995\u09b0\u09c7\u0964\n\n### \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6:\n{instruction}\n\n### \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be:"
            },
            "el": {
                "prompt_input": "\u03a0\u03b1\u03c1\u03b1\u03ba\u03ac\u03c4\u03c9 \u03b5\u03af\u03bd\u03b1\u03b9 \u03bc\u03b9\u03b1 \u03bf\u03b4\u03b7\u03b3\u03af\u03b1 \u03c0\u03bf\u03c5 \u03c0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c6\u03b5\u03b9 \u03bc\u03b9\u03b1 \u03b5\u03c1\u03b3\u03b1\u03c3\u03af\u03b1, \u03c3\u03b5 \u03c3\u03c5\u03bd\u03b4\u03c5\u03b1\u03c3\u03bc\u03cc \u03bc\u03b5 \u03bc\u03b9\u03b1 \u03b5\u03af\u03c3\u03bf\u03b4\u03bf \u03c0\u03bf\u03c5 \u03c0\u03b1\u03c1\u03ad\u03c7\u03b5\u03b9 \u03c0\u03b5\u03c1\u03b1\u03b9\u03c4\u03ad\u03c1\u03c9 \u03c0\u03bb\u03b1\u03af\u03c3\u03b9\u03bf. \u0393\u03c1\u03ac\u03c8\u03c4\u03b5 \u03bc\u03b9\u03b1 \u03b1\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7 \u03c0\u03bf\u03c5 \u03bf\u03bb\u03bf\u03ba\u03bb\u03b7\u03c1\u03ce\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03ac\u03bb\u03bb\u03b7\u03bb\u03b1 \u03c4\u03bf \u03b1\u03af\u03c4\u03b7\u03bc\u03b1.\n\n### \u039f\u03b4\u03b7\u03b3\u03af\u03b1:\n{instruction}\n\n### \u0395\u03b9\u03c3\u03b1\u03b3\u03c9\u03b3\u03ae:\n{input}\n\n### \u0391\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7:",
                "prompt_no_input": "\u03a0\u03b1\u03c1\u03b1\u03ba\u03ac\u03c4\u03c9 \u03b5\u03af\u03bd\u03b1\u03b9 \u03bc\u03b9\u03b1 \u03bf\u03b4\u03b7\u03b3\u03af\u03b1 \u03c0\u03bf\u03c5 \u03c0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c6\u03b5\u03b9 \u03bc\u03b9\u03b1 \u03b5\u03c1\u03b3\u03b1\u03c3\u03af\u03b1. \u0393\u03c1\u03ac\u03c8\u03c4\u03b5 \u03bc\u03b9\u03b1 \u03b1\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7 \u03c0\u03bf\u03c5 \u03bf\u03bb\u03bf\u03ba\u03bb\u03b7\u03c1\u03ce\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03ac\u03bb\u03bb\u03b7\u03bb\u03b1 \u03c4\u03bf \u03b1\u03af\u03c4\u03b7\u03bc\u03b1.\n\n### \u039f\u03b4\u03b7\u03b3\u03af\u03b1:\n{instruction}\n\n### \u0391\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7:"
            },
            "es": {
                "prompt_input": "A continuaci\u00f3n se muestra una instrucci\u00f3n que describe una tarea, junto con una entrada que proporciona m\u00e1s contexto. Escriba una respuesta que complete adecuadamente la solicitud.\n\n### {user}Instrucci\u00f3n:\n{instruction}\n\n### Aporte:\n{input}\n\n### {ai}Respuesta:",
                "prompt_no_input": "A continuaci\u00f3n se muestra una instrucci\u00f3n que describe una tarea. Escriba una respuesta que complete adecuadamente la solicitud.\n\n### {user}Instrucci\u00f3n:\n{instruction}\n\n### {ai}Respuesta:"
            },
            "vi": {
                "prompt_input": "D\u01b0\u1edbi \u0111\u00e2y l\u00e0 h\u01b0\u1edbng d\u1eabn m\u00f4 t\u1ea3 m\u1ed9t nhi\u1ec7m v\u1ee5, \u0111\u01b0\u1ee3c gh\u00e9p n\u1ed1i v\u1edbi \u0111\u1ea7u v\u00e0o \u0111\u1ec3 cung c\u1ea5p th\u00eam ng\u1eef c\u1ea3nh. Vi\u1ebft m\u1ed9t ph\u1ea3n h\u1ed3i ho\u00e0n th\u00e0nh y\u00eau c\u1ea7u m\u1ed9t c\u00e1ch th\u00edch h\u1ee3p.\n\n### {user}Ch\u1ec9 d\u1eabn:\n{instruction}\n\n### \u0110\u1ea7u v\u00e0o:\n{input}\n\n### {ai}Ph\u1ea3n \u1ee9ng:",
                "prompt_no_input": "D\u01b0\u1edbi \u0111\u00e2y l\u00e0 h\u01b0\u1edbng d\u1eabn m\u00f4 t\u1ea3 m\u1ed9t nhi\u1ec7m v\u1ee5. Vi\u1ebft m\u1ed9t ph\u1ea3n h\u1ed3i ho\u00e0n th\u00e0nh y\u00eau c\u1ea7u m\u1ed9t c\u00e1ch th\u00edch h\u1ee3p.\n\n### {user}Ch\u1ec9 d\u1eabn:\n{instruction}\n\n### {ai}Ph\u1ea3n \u1ee9ng:"
            },
            "id": {
                "prompt_input": "Di bawah ini adalah instruksi yang menjelaskan tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. Tulis tanggapan yang melengkapi permintaan dengan tepat.\n\n### {user}Petunjuk:\n{instruction}\n\n### Memasukkan:\n{input}\n\n### {ai}Tanggapan:",
                "prompt_no_input": "Di bawah ini adalah instruksi yang menjelaskan suatu tugas. Tulis tanggapan yang melengkapi permintaan dengan tepat.\n\n### {user}Petunjuk:\n{instruction}\n\n### {ai}Tanggapan:"
            },
            "fr": {
                "prompt_input": "Vous trouverez ci-dessous une instruction d\u00e9crivant une t\u00e2che, associ\u00e9e \u00e0 une entr\u00e9e fournissant un contexte suppl\u00e9mentaire. \u00c9crivez une r\u00e9ponse qui compl\u00e8te de mani\u00e8re appropri\u00e9e la demande.\n\n### Instruction:\n{instruction}\n\n### Saisir:\n{input}\n\n### R\u00e9ponse:",
                "prompt_no_input": "Vous trouverez ci-dessous une instruction qui d\u00e9crit une t\u00e2che. \u00c9crivez une r\u00e9ponse qui compl\u00e8te de mani\u00e8re appropri\u00e9e la demande.\n\n### Instruction:\n{instruction}\n\n### R\u00e9ponse:",
            },
            "de": {
                "prompt_input": "Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt, gepaart mit einer Eingabe, die weiteren Kontext bereitstellt. Schreiben Sie eine Antwort, die die Anfrage angemessen vervollst\u00e4ndigt.\n\n### Anweisung:\n{instruction}\n\n### Eingabe:\n{input}\n\n### Antwort:",
                "prompt_no_input": "Nachfolgend finden Sie eine Anleitung, die eine Aufgabe beschreibt. Schreiben Sie eine Antwort, die die Anfrage angemessen vervollst\u00e4ndigt.\n\n### Anweisung:\n{instruction}\n\n### Antwort:",
            },
            "zh": {
                "prompt_input": "\u4e0b\u9762\u662f\u63cf\u8ff0\u4efb\u52a1\u7684\u6307\u4ee4\uff0c\u5e76\u4e0e\u63d0\u4f9b\u8fdb\u4e00\u6b65\u4e0a\u4e0b\u6587\u7684\u8f93\u5165\u914d\u5bf9\u3002\u7f16\u5199\u9002\u5f53\u5b8c\u6210\u8bf7\u6c42\u7684\u54cd\u5e94\u3002\n\n\uff03\uff03\uff03 \u64cd\u4f5c\u8bf4\u660e\uff1a\n{instruction}\n\n\uff03\uff03\uff03 \u8f93\u5165\uff1a\n{input}\n\n\uff03\uff03\uff03 \u56de\u590d\uff1a",
                "prompt_no_input": "\u4e0b\u9762\u662f\u63cf\u8ff0\u4efb\u52a1\u7684\u6307\u4ee4\u3002\u7f16\u5199\u9002\u5f53\u5b8c\u6210\u8bf7\u6c42\u7684\u54cd\u5e94\u3002\n\n\uff03\uff03\uff03 \u64cd\u4f5c\u8bf4\u660e\uff1a\n{instruction}\n\n\uff03\uff03\uff03 \u56de\u590d\uff1a",
            },
            "ru": {
                "prompt_input": "\u041d\u0438\u0436\u0435 \u043f\u0440\u0438\u0432\u0435\u0434\u0435\u043d\u0430 \u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f, \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u044e\u0449\u0430\u044f \u0437\u0430\u0434\u0430\u0447\u0443 \u0432 \u0441\u043e\u0447\u0435\u0442\u0430\u043d\u0438\u0438 \u0441 \u0432\u0445\u043e\u0434\u043d\u044b\u043c\u0438 \u0434\u0430\u043d\u043d\u044b\u043c\u0438, \u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432\u043b\u044f\u044e\u0449\u0438\u043c\u0438 \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0439 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442. \u041d\u0430\u043f\u0438\u0448\u0438\u0442\u0435 \u043e\u0442\u0432\u0435\u0442, \u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0443\u044e\u0449\u0438\u043c \u043e\u0431\u0440\u0430\u0437\u043e\u043c \u0434\u043e\u043f\u043e\u043b\u043d\u044f\u0435\u0442 \u0437\u0430\u043f\u0440\u043e\u0441.\n\n### \u0418\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f:\n{instruction}\n\n### \u0412\u0445\u043e\u0434:\n{input}\n\n### \u041e\u0442\u0432\u0435\u0442:",
                "prompt_no_input": "\u041d\u0438\u0436\u0435 \u043f\u0440\u0438\u0432\u0435\u0434\u0435\u043d\u0430 \u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f, \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u044e\u0449\u0430\u044f \u0437\u0430\u0434\u0430\u0447\u0443. \u041d\u0430\u043f\u0438\u0448\u0438\u0442\u0435 \u043e\u0442\u0432\u0435\u0442, \u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0443\u044e\u0449\u0438\u043c \u043e\u0431\u0440\u0430\u0437\u043e\u043c \u0434\u043e\u043f\u043e\u043b\u043d\u044f\u0435\u0442 \u0437\u0430\u043f\u0440\u043e\u0441.\n\n### \u0418\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f:\n{instruction}\n\n### \u041e\u0442\u0432\u0435\u0442:",
            }
        }
        strong_user_ai_dict = {
            "es": {
                "user": "Usuario fuerte de inteligencia artificial ",
                "ai": "Inteligencia artificial fuerte "
            },
            "vi": {
                "user": "Ng\u01b0\u1eddi d\u00f9ng tr\u00ed tu\u1ec7 nh\u00e2n t\u1ea1o m\u1ea1nh ",
                "ai": "Tr\u00ed tu\u1ec7 nh\u00e2n t\u1ea1o m\u1ea1nh "
            },
            "id": {
                "user": "Pengguna kecerdasan buatan yang kuat ",
                "ai": "Kecerdasan buatan yang kuat "
            },
            "fr": {
                "user": "Fort utilisateur d\u2019intelligence artificielle ",
                "ai": "Une intelligence artificielle forte ",
            },
            "de": {
                "user": "Starker Benutzer k\u00fcnstlicher Intelligenz ",
                "ai": "Starke k\u00fcnstliche Intelligenz ",
            },
            "zh": {
                "user": "\u5f3a\u5927\u7684\u4eba\u5de5\u667a\u80fd\u7528\u6237 ",
                "ai": "\u5f3a\u5927\u7684\u4eba\u5de5\u667a\u80fd ",
            },
            "ru": {
                "user": "\u0423\u0432\u0435\u0440\u0435\u043d\u043d\u044b\u0439 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c \u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0433\u043e \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442\u0430 ",
                "ai": "\u0421\u0438\u043b\u044c\u043d\u044b\u0439 \u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442 ",
            }
        }
        weak_user_ai_dict = {
            "es": {
                "user": "Usuario d\u00e9bil de inteligencia artificial ",
                "ai": "Inteligencia artificial d\u00e9bil "
            },
            "vi": {
                "user": "Ng\u01b0\u1eddi d\u00f9ng tr\u00ed tu\u1ec7 nh\u00e2n t\u1ea1o y\u1ebfu ",
                "ai": "Tr\u00ed tu\u1ec7 nh\u00e2n t\u1ea1o y\u1ebfu "
            },
            "id": {
                "user": "Pengguna kecerdasan buatan yang lemah ",
                "ai": "Kecerdasan buatan yang lemah "
            },
            "fr": {
                "user": "Faible utilisateur de l\u2019intelligence artificielle ",
                "ai": "Faible intelligence artificielle "
            },
            "de": {
                "user": "Schwacher Benutzer k\u00fcnstlicher Intelligenz ",
                "ai": "Schwache k\u00fcnstliche Intelligenz "
            },
            "zh": {
                "user": "\u5f31\u4eba\u5de5\u667a\u80fd\u7528\u6237 ",
                "ai": "\u5f31\u4eba\u5de5\u667a\u80fd "
            },
            "ru": {
                "user": "\u0421\u043b\u0430\u0431\u044b\u0439 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c \u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0433\u043e \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442\u0430 ",
                "ai": "\u0421\u043b\u0430\u0431\u044b\u0439 \u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442 "
            }
        }

        TACO_PROMPT_DICT = {
            "prompt_input": (
                "Translate the following instructions and context from {lang} to English, formulate a response in English, "
                "and then translate that response back into {lang}."
            ),
            "response_input": (
                "### Instruction in English:\n{en_instruction}\n\n"
                "### Context in English:\n{en_input}\n\n"
                "### Response in English:\n{en_output}\n\n"
                "### Response in {lang}:\n{output}"
            ),
            "prompt_no_input": (
                "Translate the following instructions from {lang} to English, formulate a response in English, "
                "and then translate that response back into {lang}."
                "### Instruction:\n{instruction}\n\n"
            ),
            "response_no_input": (
                "### Instruction in English:\n{en_instruction}\n\n"
                "### Response in English:\n{en_output}\n\n"
                "### Response in {lang}:\n{output}"
            ),
        }

        PLUG_PROMPT_DICT = {
            "prompt_input": (
                "Please interpret the instruction and context in English, and then respond both in English and in {lang}. "
                "### Instruction:\n{instruction}\n\n"
                "### Context:\n{input}\n\n"
            ),
            "response_input": (
                "### English Instruction:\n{en_instruction}\n\n"
                "### English Context:\n{en_input}\n\n"
                "### English Response:\n{en_output}\n\n"
                "### Response:\n{output}"
            ),
            "prompt_no_input": (
                "Please interpret the instruction in English, and then respond both in English and in {lang}. "
                "### Instruction:\n{instruction}\n\n"
            ),
            "response_no_input": (
                "### English Instruction:\n{en_instruction}\n\n"
                "### English Response:\n{en_output}\n\n"
                "### Response:\n{output}"
            ),
        }

        if args.method == 'plug':
            prompt_dict = PLUG_PROMPT_DICT
            prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]
            example = {k: v for k, v in example.items()}
            example['lang'] = args.alpaca_lang
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            ]
            targets = [
                ""
            ]

        else:
            prompt_dict = lang_to_prompt_dict[args.alpaca_lang]
            example = {k: v for k, v in example.items()}

            prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            ]
            targets = [""]
        if output_prompt_only:
            return sources
        else:
            raise NotImplementedError

    import json
    lines = open(args.input_file).readlines()
    start_i = int(len(lines) * args.split_num / args.total_splits)
    end_i = int(len(lines) * (1 + args.split_num) / args.total_splits)
    lines = lines[start_i:end_i]
    input_sentences = sum([tokenize_alpaca(json.loads(line), output_prompt_only=True) for line in lines], [])

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=input_sentences,
                                  prompt_template=prompt_template,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id,
                                  num_prepend_vtokens=args.num_prepend_vtokens,
                                  model_name=model_name)
    input_lengths = [x.size(0) for x in batch_input_ids]

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    runner_cls = ModelRunner
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
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
    print(runner_kwargs)
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
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            lora_uids=args.lora_task_uids,
            prompt_table_path=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            return_dict=True)
        torch.cuda.synchronize()

    if args.streaming:
        for curr_outputs in throttle_generator(outputs,
                                               args.streaming_interval):
            if runtime_rank == 0:
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                print_output(tokenizer,
                             output_ids,
                             input_lengths,
                             sequence_lengths,
                             output_csv=args.output_csv,
                             output_npy=args.output_npy)
    else:
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']
            context_logits = None
            generation_logits = None
            if runner.gather_all_token_logits:
                context_logits = outputs['context_logits']
                generation_logits = outputs['generation_logits']
            print_output(tokenizer,
                         output_ids,
                         input_lengths,
                         sequence_lengths,
                         output_csv=args.output_csv,
                         output_npy=args.output_npy,
                         context_logits=context_logits,
                         generation_logits=generation_logits,
                         output_logits_npy=args.output_logits_npy)


if __name__ == '__main__':
    args = parse_arguments()
    output_file = open(args.output_file, 'w')
    main(args)