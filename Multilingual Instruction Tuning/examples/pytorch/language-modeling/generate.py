import os
import sys
import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, GenerationConfig, LlamaForCausalLM
from peft import PeftModel

from utils.callbacks import Iteratorize, Stream
from .prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--base_model", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--lora_weights", type=str, default="", required=True)
    parser.add_argument("--template_dir", type=str, default="./templates")
    parser.add_argument("--prompt_template_name", type=str, default="bactrian")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert (args.base_model), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(args.prompt_template_name, args.template_dir)
    # Todo: better handle
    tokenizer_class = LlamaTokenizer if 'llama' in args.base_model else AutoTokenizer
    model_class = LlamaForCausalLM if 'llama' in args.base_model else AutoModelForCausalLM

    tokenizer = tokenizer_class.from_pretrained(args.base_model)
    if device == "cuda":
        model = model_class.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = model_class.from_pretrained(
            args.base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = model_class.from_pretrained(
            args.base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # if config.model_type == "qwen":
        #     # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
        #     tokenizer.pad_token = "<|endoftext|>"
        # else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # if 'llama' in args.base_model:
    #     model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #     model.config.bos_token_id = 1
    #     model.config.eos_token_id = 2
    #
    # if not args.load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)


if __name__ == "__main__":
    main()