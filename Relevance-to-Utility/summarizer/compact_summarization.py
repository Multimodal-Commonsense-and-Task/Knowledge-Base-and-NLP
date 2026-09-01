import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from utils import create_prompt, parse_output_without_sentence
import os
from tqdm import tqdm
import argparse
import re


def create_prompt(example, iteration, iter_idx, document_input, prev_summary, prev_eval, tokenizer, eos_token="<|endoftext|>", add_generation_prompt=False):
    if iter_idx == 0:
        instruction = "1. Generate a summary of source documents to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."

        prompt = f"{instruction}\n\nQuestion: {example['question']}\n\nSource documents: {document_input}\n\nSummary:"
    else:
        instruction = "1. Generate a summary of the previous summary and the source documents to answer the question based on the evaluation of the previous summary. The evaluation indicates the missing information needed to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."

        prompt = f"{instruction}\n\nQuestion: {example['question']}\n\nPrevious summary: {prev_summary}\n\nEvaluation of previous summary: {prev_eval}\n\nSource documents: {document_input}\n\nSummary:"

    messages = [
        {"role": "user", "content": prompt},
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


    return chat_format


def parse_output_without_sentence(text):
    sections = {}

    summary_pattern_with_prefix = r'(Summary:)(.*?)(?=Evaluation:|$)'
    summary_pattern_without_prefix = r'(^.*?)(?=Evaluation:|$)'
    evaluation_pattern = r'(Evaluation:)(.*?)(?=Summary:|$)'
    
    # Find all matches for each section
    summary_match_with_prefix = re.search(summary_pattern_with_prefix, text, re.DOTALL)
    summary_match_without_prefix = re.search(summary_pattern_without_prefix, text, re.DOTALL)
    evaluation_match = re.search(evaluation_pattern, text, re.DOTALL)
    
   # Extracting and cleaning the matched content
    if summary_match_with_prefix:
        sections['summary'] = summary_match_with_prefix.group(2).strip()
    elif summary_match_without_prefix:
        sections['summary'] = summary_match_without_prefix.group(1).strip()
    else:
        sections['summary'] = ""

    if evaluation_match:
        sections['eval'] = evaluation_match.group(2).strip()

    # Cleaning extra newlines if necessary
    sections['summary'] = sections['summary'].replace("\n\n", "")
    sections['eval'] = sections['eval'].replace("\n\n", "")

    return sections


# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Generate document summaries using a pre-trained model.")
parser.add_argument('--model_dir', type=str, default='models/CompAct-7b', help='Directory of the pre-trained model')
parser.add_argument('--search_cache_path', type=str, default='cache/hotpotqa/search_cache_b500_c500.json', help='Path to the search cache file')
parser.add_argument('--window_size', type=int, default=5, help='Number of documents to process in each iteration')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = args.model_dir
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
WINDOW_SIZE = args.window_size
# example = json.load(open('./data/example.json')) # example case with retrieved documents
# print(f"question: {example['question']}\nanswer: {example['answer']}")

search_cache_path = args.search_cache_path
# Load existing caches or initialize empty dictionaries
print(f"#> search_cache_path: {search_cache_path}")
if os.path.exists(search_cache_path):
    with open(search_cache_path, 'r', encoding='utf-8') as f:
        search_cache = json.load(f)
else:
    raise FileNotFoundError(f"Search cache file not found: {search_cache_path}")

# convert to list of documents



for question, docs in tqdm(search_cache.items(), desc="Loading Search Cache"):
    example = {}
    prev_summary = []
    prev_eval = []
    example['question'] = question
    documents = [doc['contents'] for doc in docs]
    for i in range(0, len(documents), WINDOW_SIZE):
        document_input = documents[i:i + WINDOW_SIZE]
        iteration = {
            "documents_list": document_input,
        }
        document_input = "\n".join(document_input)
        # # actively compress documents until it finds all necessary evidence
        # for i, iteration in enumerate(example['iterations']):
        #     segment = iteration['documents_list']
        #     document_input = "\n".join(segment)

        # load previous output
        prev_summary_temp = prev_summary[-1] if i != 0 else ""
        prev_eval_temp = prev_eval[-1].replace('[INCOMPLETE]', '').strip() if i != 0 else ""

        # create prompt
        input_prompt = create_prompt(example, iteration, i, document_input, prev_summary_temp, prev_eval_temp, tokenizer, eos_token="", add_generation_prompt=True)
        
        # compress
        with torch.no_grad():
            inputs = tokenizer(input_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=500, temperature=0, top_p=1.0, pad_token_id=tokenizer.eos_token_id)
        iteration['output'] = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()

        # parsing
        parsed_sections = parse_output_without_sentence(iteration['output'])
        prev_summary.append(parsed_sections['summary'])
        prev_eval.append(parsed_sections['eval'])

        # early termination
        if "[COMPLETE]" in parsed_sections['eval']:
            break
    new_docs = []
    new_docs.append({
        "id": docs[0]['id'],
        "title": '',
        "contents": prev_summary[-1],
        "eval": prev_eval[-1],
    })
    search_cache[question] = new_docs
# Save the updated search cache
with open(search_cache_path.replace('.json', '_compact.json'), 'w', encoding='utf-8') as f:
    json.dump(search_cache, f, ensure_ascii=False, indent=4)