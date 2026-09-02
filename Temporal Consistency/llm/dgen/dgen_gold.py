from langchain_core.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
import argparse
import json
from tqdm import tqdm
from langchain.output_parsers import RetryOutputParser
import datetime
# os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6,7"

data = []
with open("dataset/test.hard.concat.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

# if targets is in the paragraph, select it.


with open("dataset/test.hard.short.gold.json", "w") as f:
    for d in tqdm(data):
        new_paragraphs = []
        paragraphs = d["paragraphs"] # list of dictionary, with key: title, text
        targets = d["targets"]
        new_paragraphs = [p for p in paragraphs if any([target.lower() in p["text"].lower() for target in targets])]
        d["context"] = "\n".join([p["text"] for p in new_paragraphs])
        f.write(json.dumps(d) + "\n")

