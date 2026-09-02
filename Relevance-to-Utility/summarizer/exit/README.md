# EXIT: Context-Aware Extractive Compression for RAG 🚀

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2412.12559-b31b1b.svg)](https://arxiv.org/abs/2412.12559)

Official implementation of "EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation"

## Overview 📋

EXIT is a context-aware extractive compression framework that improves both the effectiveness and efficiency of Retrieval-Augmented Generation (RAG) by:

- 🎯 Preserving critical information while reducing context size
- 🔍 Considering full document context when evaluating sentence importance
- ⚡ Enabling parallelizable, context-aware extraction
- 🎚️ Adapting dynamically to query complexity
- ⚖️ Balancing compression ratio and answer accuracy

## Installation 💻

```bash
# Clone the repository
git clone https://github.com/ThisIsHwang/EXIT.git
cd EXIT

# Create a new conda environment
conda create -n exit python=3.8
conda activate exit

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quickstart 🚀

Here's a simple example demonstrating the EXIT RAG pipeline:

```python
from exit_rag import ExitRAG, Document

# Initialize pipeline
rag = ExitRAG(
    retriever_model="google/gemma-2b-it",
    compression_model="doubleyyh/exit-gemma-2b",
    reader_model="meta-llama/Llama-3.1-8B-Instruct"
)

# Example query and document
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

# Run RAG pipeline with compression
result = rag.run_rag(query, documents)

# Print results
print("\nQuery:", result["query"])
print("\nCompressed Context:", result["compressed_context"])
print("\nAnswer:", result["answer"])
print(f"\nGeneration Time: {result['generation_time']:.2f}s")
```
## Data Preparation 📚

### Download Datasets

You can download the evaluation datasets (NaturalQuestions, TriviaQA, HotpotQA, 2WikiMultiHopQA) from the [CompAct repository](https://github.com/dmis-lab/CompAct).

### Dataset Structure

Each dataset follows the format:

```json

{

    "question": "How do solid-state drives improve computer performance?",

    "ctxs": [

        {

            "title": "Document Title",

            "text": "Document content...",

            "score": 1.0

        },

        ...

    ]

}

```

## Model Details 🔧

- **Base Model**: Gemma-2b-it
- **Training Method**: PEFT/LoRA
- **Training Data**: HotpotQA dataset with:
  - Positive examples: Sentences marked as supporting facts
  - Hard negatives: Sentences from same documents but not supporting facts
  - Random negatives: Sentences from unrelated documents
- **Recommended Parameters**:
  - Compression threshold (tau): 0.5
  - Cache directory: Configurable via initialization

## Key Features 🌟

### Document Compression

```python
compressed_text, selections, scores = rag.compress_documents(
    query=query,
    documents=documents,
    threshold=0.5  # Adjustable compression threshold
)
```

### Answer Generation

```python
answer, generation_time = rag.generate_answer(
    query=query,
    context=compressed_text
)
```

### Complete RAG Pipeline

```python
result = rag.run_rag(
    query=query,
    documents=documents,
    compression_threshold=0.5
)
```

## Performance 📊

EXIT demonstrates superior performance in:
- Token count reduction
- Answer accuracy preservation
- End-to-end latency reduction
- Multi-hop question handling

## Limitations ⚠️

- Currently optimized for English text only
- No support for cross-lingual compression
- Requires GPU for optimal performance

## Citation 📚

If you use EXIT in your research, please cite our paper:

```bibtex
@article{hwang2024exit,
  title={EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation},
  author={Hwang, Taeho and Cho, Sukmin and Jeong, Soyeong and Song, Hoyun and Han, SeungYoon and Park, Jong C.},
  journal={arXiv preprint arXiv:2412.12559},
  year={2024}
}
```

## License 📄

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact 📧

For questions or issues:
- Open an issue in this repository
- Contact: doubleyyh@kaist.ac.kr