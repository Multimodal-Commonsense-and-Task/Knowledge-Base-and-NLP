from call_llm import LLM
from hamu_tool.dataset import DataLoader
from tqdm import tqdm

dataset = input('Dataset: ')

with open(f'data/{dataset}/prompt.generate.keyword.txt', 'r') as file:
    prompt_base = file.read().strip()

loader = DataLoader.load(f'beir/{dataset}')
llm = LLM(client_type='openai')

keywords_set = set()
for doc in tqdm(loader.get_docs(mode='test'), total=loader.total_docs(mode='test'), desc=f'[{dataset}] Generating keywords'):
    prompt = prompt_base.replace('##document##', doc.text)
    keywords = llm(prompt=prompt)[0]
    keywords = keywords.lower().replace('keywords-4:', '').strip().split(',')
    keywords = [keyword.strip() for keyword in keywords]
    keywords_set.update(keywords)

with open(f'data/{dataset}/keywords.txt', 'w') as file:
    for keyword in keywords_set:
        file.write(f'{keyword}\n')
