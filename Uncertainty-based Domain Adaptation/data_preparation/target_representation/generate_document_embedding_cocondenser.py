import json
import torch
import argparse
from tqdm import tqdm
from functools import partial
from transformers import AutoModel, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def encode_text(model, max_seq_length, tokenizer, text):
    """
    Encodes input text using a coCondenser model.
    We use the [CLS] embedding from the last hidden state (index [0]).
    """
    inputs = tokenizer.batch_encode_plus(
        text,
        truncation="longest_first",
        max_length=max_seq_length,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # coCondenser doesn't provide a pooler_output for retrieval,
    # so we take the [CLS] embedding directly from last_hidden_state.
    cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
    return cls_embeddings

def main_run(collection_data_filepath,
             save_collection_embedding_filepath,
             cache_dir,
             batch_size=32,
             encoder_model_name_or_path='Luyu/co-condenser-marco',
             max_seq_length=512):
    # 1. Load collection documents
    docid_doctext_list = []
    with open(collection_data_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            docid_doctext_list.append(
                {'docid': data['docid'], 'doctext': data['doctext']}
            )

    # 2. Load coCondenser model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name_or_path,
                                              cache_dir=cache_dir)
    model = AutoModel.from_pretrained(encoder_model_name_or_path,
                                      cache_dir=cache_dir)
    model.to(device)
    print("... coCondenser Model Loaded into GPU")

    text_encoder = partial(encode_text, model, max_seq_length, tokenizer)

    # 3. Encode document text
    dataset_doc_emb_dict = {}
    for i in tqdm(range(0, len(docid_doctext_list), batch_size)):
        batch_data = docid_doctext_list[i : i + batch_size]

        batch_doc_text = [e['doctext'] for e in batch_data]
        batch_doc_ids  = [e['docid']   for e in batch_data]

        # Get coCondenser [CLS] embeddings
        text_emb = text_encoder(batch_doc_text)

        # Store embeddings in a dict
        for doc_id, emb in zip(batch_doc_ids, text_emb):
            dataset_doc_emb_dict[str(doc_id)] = emb

    # 4. Save document embeddings
    torch.save(dataset_doc_emb_dict, save_collection_embedding_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Encoding using coCondenser')
    parser.add_argument('--collection_data_filepath', required=True, type=str,
                        help='Path to document collection JSON lines')
    parser.add_argument('--save_collection_embedding_filepath', required=True, type=str,
                        help='Output path for document embeddings')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='Cache directory for downloading models/tokenizers')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='Batch size for inference')
    parser.add_argument('--encoder_model_name_or_path', type=str,
                        default='Luyu/co-condenser-marco-retriever',
                        required=False,
                        help='coCondenser model name or path')
    parser.add_argument('--max_seq_length', type=int, default=512, required=False,
                        help='Maximum sequence length to encode text')
    args = parser.parse_args()

    main_run(args.collection_data_filepath,
             args.save_collection_embedding_filepath,
             args.cache_dir,
             args.batch_size,
             args.encoder_model_name_or_path,
             args.max_seq_length)
