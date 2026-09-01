from transformers import LlamaForCausalLM, set_seed, AutoConfig, QuantizedCacheConfig
import torch
model = '../Meta-Llama-3.1-8B-Instruct'

load_in_4bit = False
# orig_model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map='cuda', load_in_4bit=load_in_4bit, attn_implementation='sdpa')
# context_len = 2 ** 10

# while True:
#     set_seed(42)
#     rand_input = torch.randint(300, 30000, [1, context_len], device='cuda')
#     orig_model.generate(rand_input, max_new_tokens=2)
#     print(context_len, torch.cuda.max_memory_allocated())
#     context_len *= 2

kv_svd_path = '../Meta-Llama-3.1-8B-Instruct_-1'
kv_keep_ratio = 0.3125
keep_initial_offsets = 4
lowrank_kv_ratio = 0.75
kv_cache_quant = "HQQ"
kv_cache_bits = 4
kv_q_group_size = 16
lowrank_kv_cache_bits = 2
lowrank_kv_q_group_size = 16
axis_key = 1
axis_value = 1

config = AutoConfig.from_pretrained(model)
config.keep_initial_offsets = keep_initial_offsets
config.lowrank_kv_ratio = lowrank_kv_ratio
from lm_eval.models.teal.modeling_llama_moremore_recent import LlamaForCausalLM, use_flashv2
# use_flashv2()
orig_model = LlamaForCausalLM.from_pretrained(model, config=config, torch_dtype=torch.float16, device_map='cuda', attn_implementation='sdpa', load_in_4bit=load_in_4bit)

if lowrank_kv_cache_bits is None:
    lowrank_kv_cache_bits = kv_cache_bits
    lowrank_kv_q_group_size = kv_q_group_size

if kv_cache_quant:
    print(f"USING lowrank kv as {lowrank_kv_cache_bits}")
    cache_config = {"nbits": lowrank_kv_cache_bits, "backend": kv_cache_quant,
                    "device": next(orig_model.parameters()).device,
                    "compute_dtype": next(orig_model.parameters()).dtype,
                    "q_group_size": lowrank_kv_q_group_size,
                    "axis_key": axis_key,
                    "axis_value": axis_value,
                    }

orig_model.replace_kv_to_lowranks(kv_svd_path, kv_keep_ratio)
orig_model.config.cache_config = None
if kv_cache_quant:
    orig_model.generation_config.update(cache_implementation="quantized", cache_config=cache_config)
    print(f"USING full kv as {kv_cache_bits}")
    cache_config = {"nbits": kv_cache_bits, "backend": kv_cache_quant,
                    "device": next(orig_model.parameters()).device,
                    "compute_dtype": next(orig_model.parameters()).dtype,
                    "q_group_size": kv_q_group_size,
                    "axis_key": axis_key,
                    "axis_value": axis_value,
                    }
    orig_model.config.cache_config = QuantizedCacheConfig.from_dict(cache_config)

context_len = 2 ** 10

while True:
    set_seed(42)
    rand_input = torch.randint(300, 30000, [1, context_len], device='cuda')
    orig_model.generate(rand_input, max_new_tokens=2)
    print(context_len, torch.cuda.max_memory_allocated())
    context_len *= 2
