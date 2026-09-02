# [NAACL 2025] PROM: Pivoted and Regulated Optimization for Multilingual Instruction Learning

This code is based on [transformers](https://github.com/huggingface/transformers).

## Running PROM
```bash
export output_dir=OUTPUT_DIR
export train_file=TRAIN_FILE
export n=23
export alpha=0.1
export lambda=0.1
accelerate launch examples/pytorch/language-modeling/run_clm_instruction_alpaca_diffhead.py \
--train_file=$train_file \
--output_dir=$output_dir \
--model_name_or_path=meta-llama/Llama-2-7b-hf\
--intermediate_id=$n \
--label_smooth=$alpha \
--loss_w=$lambda
```

## Running xLLaMA2
```bash
export output_dir=OUTPUT_DIR
export train_file=TRAIN_FILE
accelerate launch examples/pytorch/language-modeling/run_clm_instruction_alpaca.py \
--train_file=$train_file \
--output_dir=$output_dir \
--model_name_or_path=meta-llama/Llama-2-7b-hf
```

## Running Bactrian+
```bash
export output_dir=OUTPUT_DIR
export train_file=TRAIN_FILE
export n=23
export alpha=0.1
export lambda=0
accelerate launch examples/pytorch/language-modeling/run_clm_instruction_alpaca_diffhead.py \
--train_file=$train_file \
--output_dir=$output_dir \
--model_name_or_path=meta-llama/Llama-2-7b-hf\
--intermediate_id=$n \
--label_smooth=$alpha \
--loss_w=$lambda
```

## Acknowledgements

This research was partially supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2025-2020-0-01789) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation), MSIT/IITP grant (2022-0-00995, 2022-0-00077/RS-2022-II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
