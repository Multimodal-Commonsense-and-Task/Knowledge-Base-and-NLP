from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel
import torch
import argparse
from tqdm import tqdm
import os
import json


def process_batch(batch_items, tokenizer, model, template, batch_size=8):
    """
    Process a batch of questions and documents
    
    Args:
        batch_items: List of (question, docs) tuples
        tokenizer: The tokenizer instance
        model: The model instance
        template: The prompt template
        batch_size: Number of items to process in each batch
    
    Returns:
        List of processed outputs
    """
    questions, docs_list = zip(*batch_items)
    
    # Prepare all prompts
    prompts = []
    for question, docs in zip(questions, docs_list):
        document_input = ""
        for doc in docs:
            title = doc.get('title', '')
            if title:
                document_input += f"## {title}\n"
            contents = doc.get('contents', '')
            if contents:
                document_input += f"{contents}\n---\n"
        
        prompt = template.format(question=question, context=document_input)
        prompts.append(prompt)
    
    # Tokenize all prompts
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        padding="longest",
        padding_side="left"
    )
    
    # Generate predictions for the entire batch
    with torch.no_grad():  # Save memory during inference
        preds = model.generate(
            **inputs.to(model.device),
            top_p=1,
            temperature=None,
            do_sample=False,
            max_new_tokens=2048,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id  # Handle padding properly
        )
    
    # Decode all predictions
    pred_token_ids = preds.sequences[:, inputs.input_ids.shape[1]:]
    pred_texts = tokenizer.batch_decode(pred_token_ids, skip_special_tokens=True)
    
    return pred_texts

# Main batchified processing loop
def batchify_processing(search_cache, tokenizer, model, template, batch_size=8):
    """
    Process the search cache in batches
    
    Args:
        search_cache: Dictionary with questions as keys and docs as values
        tokenizer: The tokenizer instance
        model: The model instance
        template: The prompt template (TEMPLATE)
        batch_size: Number of questions to process in each batch
    """
    # Convert search_cache items to list for batching
    cache_items = list(search_cache.items())
    
    # Process in batches
    for i in tqdm(range(0, len(cache_items), batch_size), desc="Processing batches"):
        batch_items = cache_items[i:i + batch_size]
        
        # Process the current batch
        outputs = process_batch(batch_items, tokenizer, model, template, batch_size)
        
        # Update search_cache with results
        for (question, docs), output in zip(batch_items, outputs):
            new_docs = [{
                "id": docs[0]['id'],
                "title": '',
                "contents": output.strip(),
            }]
            search_cache[question] = new_docs
    return search_cache



# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Generate document summaries using a pre-trained model.")
parser.add_argument('--base_model', type=str, default='models/Llama-2-7b-chat-hf', help='Directory of the pre-trained model')
parser.add_argument('--adapter', type=str, default='models/Refiner-7B', help='Directory of the adapter model')
parser.add_argument('--search_cache_path', type=str, default='cache/hotpotqa/search_cache_b500_c500.json', help='Path to the search cache file')
parser.add_argument('--batch_size', type=int, default=1, help='Number of batch to process in each iteration')
args = parser.parse_args()

for k, v in vars(args).items():
    print(f"#> {k}: {v}")

base_model = args.base_model
adapter = args.adapter
tokenizer = AutoTokenizer.from_pretrained(base_model)
base_model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
model.eval()

TEMPLATE = "[INST]<<SYS>>[MONITOR]{context}<</SYS>>{question}[/INST] "

search_cache_path = args.search_cache_path
# Load existing caches or initialize empty dictionaries
print(f"#> search_cache_path: {search_cache_path}")
if os.path.exists(search_cache_path):
    with open(search_cache_path, 'r', encoding='utf-8') as f:
        search_cache = json.load(f)
else:
    raise FileNotFoundError(f"Search cache file not found: {search_cache_path}")

# Process the search cache in batches
# batchify_processing(search_cache, tokenizer, model, TEMPLATE, batch_size=args.batch_size)
search_cache = batchify_processing(
    search_cache, tokenizer, model, TEMPLATE, batch_size=args.batch_size
)

print(f"#> Processed {len(search_cache)} questions in the search cache. Saved to {search_cache_path.replace('.json', '_refiner_rag.json')}")
with open(search_cache_path.replace('.json', '_refiner_rag.json'), 'w', encoding='utf-8') as f:
    json.dump(search_cache, f, ensure_ascii=False, indent=4)

