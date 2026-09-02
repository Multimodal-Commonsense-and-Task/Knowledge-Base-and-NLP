import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from harp import HARPWrapper
from generation import autoregressive_generate, GreedyProcessor


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    argparser.add_argument("--quantize", type=str, default=None)
    argparser.add_argument("--device_map", type=str, default="cuda")
    argparser.add_argument("--theta", type=float, default=1.0)
    argparser.add_argument("--delta", type=float, default=0.2)
    argparser.add_argument("--beta", type=float, default=0.5)
    argparser.add_argument("--seed", type=int, default=3)
    argparser.add_argument("--max_gen_len", type=int, default=120)
    argparser.add_argument("--not_instruct", action="store_true")
    argparser.add_argument("--eos_tokens", type=str, nargs="+", default="<|eot_id|>")
    args = argparser.parse_args()

    print(f"Loading model \033[32m{args.model}\033[0m...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.quantize is not None:
        print(f"Quantizing model in \033[32m{args.quantize}\033[0m...")
        config = QuantoConfig(weights=args.quantize)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=config,
            device_map=args.device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map=args.device_map, trust_remote_code=True
        )

    model = HARPWrapper(model, theta=args.theta, delta=args.delta, beta=args.beta)
    logits_processor = GreedyProcessor()
    eos_tokens_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [0]
    if not args.not_instruct:
        eos_tokens_ids += [tokenizer.convert_tokens_to_ids(token) for token in args.eos_tokens]

    while True:
        prompt = input("\033[34m> \033[0m")
        if not args.not_instruct:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        tokenized_input = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

        print("\033[44mVanilla Generation...\033[0m")
        output = autoregressive_generate(
            inputs=tokenized_input,
            model=model,
            max_gen_len=args.max_gen_len,
            logits_processor=logits_processor,
            eos_tokens_id=eos_tokens_ids,
            vanilla=True,
        )
        print(tokenizer.decode(output))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

        print("\033[41mHARP Generation...\033[0m")
        output = autoregressive_generate(
            inputs=tokenized_input,
            model=model,
            max_gen_len=args.max_gen_len,
            logits_processor=logits_processor,
            eos_tokens_id=eos_tokens_ids,
            vanilla=False,
        )
        print(tokenizer.decode(output))


if __name__ == "__main__":
    main()
