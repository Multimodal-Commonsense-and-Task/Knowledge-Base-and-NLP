from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--orig_model', default='/data/open-lm-private_443/llama-hf/llama-2-70b-hf')
parser.add_argument('--load_peft', default=None)
parser.add_argument('--save_path', default=None)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.orig_model)
model = PeftModel.from_pretrained(model, args.load_peft)
model = model.merge_and_unload()
model = model.bfloat16()
model.save_pretrained(args.save_path)

tokenizer = AutoTokenizer.from_pretrained(args.orig_model)
tokenizer.save_pretrained(args.save_path)