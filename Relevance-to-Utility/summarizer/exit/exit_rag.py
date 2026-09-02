#!/usr/bin/env python3
"""
EXIT RAG Pipeline Quickstart
This script demonstrates an end-to-end RAG pipeline using EXIT for context compression.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import spacy
from typing import List, Tuple
import time
from dataclasses import dataclass
from fvcore.nn import FlopCountAnalysis

@dataclass
class Document:
    """Container for document content"""
    title: str
    text: str
    score: float = 1.0

class ExitRAG:
    """END-to-END Retrieval-Augmented Generation with EXIT compression"""
    
    def __init__(
        self,
        retriever_model: str = "google/gemma-2b-it",
        compression_model: str = "doubleyyh/exit-gemma-2b",
        reader_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda"
    ):
        # Initialize models
        print("Loading models...")
        
        # Initialize EXIT compression model
        base_model = AutoModelForCausalLM.from_pretrained(
            retriever_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.exit_model = PeftModel.from_pretrained(base_model, compression_model)
        self.exit_tokenizer = AutoTokenizer.from_pretrained(retriever_model)
        
        # Initialize reader model
        # self.reader = AutoModelForCausalLM.from_pretrained(
        #     reader_model, 
        #     device_map="auto",
        # )
        # self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_model)
        
        # Initialize sentence splitter
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")
        
        self.device = device

    def get_sentence_relevance(
        self,
        query: str,
        context: str,
        sentence: str,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """Determine if a sentence is relevant using EXIT model"""
        
        prompt = f'''<start_of_turn>user
Query:
{query}
Full context:
{context}
Sentence:
{sentence}
Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>
<start_of_turn>model
'''
        inputs = self.exit_tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.exit_model.device)


        
        with torch.no_grad():
            outputs = self.exit_model(**inputs)
            yes_id = self.exit_tokenizer.encode("Yes", add_special_tokens=False)
            no_id = self.exit_tokenizer.encode("No", add_special_tokens=False)
            logits = outputs.logits[0, -1, [yes_id[0], no_id[0]]]
            prob = torch.softmax(logits, dim=0)[0].item()

        return prob >= threshold, prob

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.5
    ) -> Tuple[str, List[bool], List[float]]:
        """Compress documents using EXIT model"""
        
        start_time = time.time()
        
        # Split documents into sentences
        all_sentences = []
        sentence_map = []  # Track which document each sentence comes from
        
        for doc_idx, doc in enumerate(documents):
            # Combine title and text
            full_text = f"{doc.title}\n{doc.text}" if doc.title else doc.text
            sentences = [sent.text.strip() for sent in self.nlp(full_text).sents]
            all_sentences.extend(sentences)
            sentence_map.extend([doc_idx] * len(sentences))
        
        # Get relevance scores for all sentences
        selected_sentences = []
        relevance_scores = []
        selections = []
        
        for sent in all_sentences:
            is_relevant, score = self.get_sentence_relevance(
                query,
                " ".join(all_sentences),  # Full context
                sent,
                threshold
            )
            selections.append(is_relevant)
            relevance_scores.append(score)
            if is_relevant:
                selected_sentences.append(sent)
        
        compressed_text = " ".join(selected_sentences)
        
        compression_time = time.time() - start_time
        print(f"Compression time: {compression_time:.2f}s")
        print(f"Compressed {len(selected_sentences)}/{len(all_sentences)} sentences")
        
        return compressed_text, selections, relevance_scores

    def generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """Generate answer using compressed context"""
        
        start_time = time.time()
        
        # Format prompt
        chat = [{
            "role": "system",
            "content": f"Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query. Do not provide any explanation."
        },
        {
            "role": "user",
            "content": f"Query: {query}\nAnswer: "
        }]
        
        prompt = self.reader_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate answer
        inputs = self.reader_tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reader.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                pad_token_id=self.reader_tokenizer.eos_token_id,
                do_sample=False
            )
            
        answer = self.reader_tokenizer.decode(
            outputs[0][inputs.input_ids.size(1):],
            skip_special_tokens=True
        ).strip()
        
        generation_time = time.time() - start_time
        
        return answer, generation_time

    def run_rag(
        self,
        query: str,
        documents: List[Document],
        compression_threshold: float = 0.5
    ) -> dict:
        """Run complete RAG pipeline with compression"""
        
        # 1. Compress documents
        compressed_text, selections, scores = self.compress_documents(
            query,
            documents,
            compression_threshold
        )
        
        # 2. Generate answer
        answer, generation_time = self.generate_answer(query, compressed_text)
        
        return {
            "query": query,
            "compressed_context": compressed_text,
            "answer": answer,
            "sentence_selections": selections,
            "relevance_scores": scores,
            "generation_time": generation_time
        }

def main():
    """Demonstrate usage of EXIT RAG pipeline"""
    
    # Initialize pipeline
    rag = ExitRAG()
    
    # Example query and documents
    query = "How do solid-state drives (SSDs) improve computer performance?"
    documents = [Document(
        title="Computer Storage Technologies",
        text="""
        Solid-state drives use flash memory to store data without moving parts.
        Unlike traditional hard drives, SSDs have no mechanical components.
        The absence of physical movement allows for much faster data access speeds.
        I bought my computer last week.
        SSDs significantly reduce boot times and application loading speeds.
        They consume less power and are more reliable than mechanical drives.
        The price of SSDs has decreased significantly in recent years.
        """
    )]
    
    # Run pipeline
    result = rag.run_rag(query, documents)
    
    # Print results
    print("\nQuery:", result["query"])
    print("\nCompressed Context:", result["compressed_context"])
    print("\nAnswer:", result["answer"])
    print(f"\nGeneration Time: {result['generation_time']:.2f}s")

if __name__ == "__main__":
    main()