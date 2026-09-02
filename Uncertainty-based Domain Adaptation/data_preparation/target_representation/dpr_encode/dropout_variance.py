#!/usr/bin/env python3
"""
mc_dropout_variance.py – Epistemic uncertainty for any SentenceTransformer model
(Fully Optimized Version)
"""

import os, json, argparse, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
torch.manual_seed(42)                  # reproducibility

def _tokenize_batch(tokenizer, texts, device):
    """Return tokenized features dict for a batch of texts."""
    encoded = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    
    features = {
        "input_ids":      encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    if "token_type_ids" in encoded:
        features["token_type_ids"] = encoded["token_type_ids"]
    return features


def _forward_pass_dropout(model, features):
    """Return [B, d] sentence embeddings from tokenized features, with dropout enabled."""
    with torch.no_grad():
        model.train()                  # keep dropout active!
        out = model(features)          # returns dict with key 'sentence_embedding'
    return out["sentence_embedding"]   # [B, d]


def mc_dropout_variance_optimized(texts, model, tokenizer, T=30, device="cuda"):
    """
    Returns a tensor [B] of scalar variances.
    Optimized by tokenizing once and repeating tensors on the GPU.
    """
    B = len(texts)
    d = model.get_sentence_embedding_dimension()

    # 1. 텍스트를 단 한 번만 토큰화 (CPU -> GPU)
    features = _tokenize_batch(tokenizer, texts, device)

    # 2. 토큰화된 텐서들을 T번 반복하여 거대 배치 생성 (GPU 상에서 수행)
    repeated_features = {}
    for key, tensor in features.items():
        # tensor shape: [B, seq_len]
        # B차원을 T번 반복 복사 (interleave)
        # 예: [1, 2] -> [1, 1, 1, 2, 2, 2] (T=3일 때)
        repeated_features[key] = tensor.repeat_interleave(T, dim=0)
        # 결과 shape: [B * T, seq_len]

    # 3. 거대 텐서 배치로 단 한 번의 순방향 패스 실행
    all_embeddings = _forward_pass_dropout(model, repeated_features) # Shape: [B * T, d]

    # 4. 결과를 [B, T, d] 형태로 재구성
    reshaped_embeddings = all_embeddings.view(B, T, d)

    # 5. T 차원에 대해 분산을 직접 계산
    element_wise_var = torch.var(reshaped_embeddings, dim=1, unbiased=True) # Shape: [B, d]

    # 6. 차원(d)에 대해 평균내어 스칼라 분산 값으로 변환
    var_scalar = element_wise_var.mean(dim=1) # Shape: [B]

    return var_scalar.cpu()


# ────────────────────────────────────────────────────────────────
# Main 함수는 변경할 필요 없음 (올바른 optimized 함수를 호출하고 있음)
# ────────────────────────────────────────────────────────────────
def build_variance_dict(base_modelname_or_path,
                        clean_document_path,
                        variance_dict_file,
                        cache_dir,
                        T=30,
                        batch_size=128,
                        device="cuda"):

    # ... (내용 동일) ...
    print("[*] loading model:", base_modelname_or_path)
    model = SentenceTransformer(base_modelname_or_path,
                                cache_folder=cache_dir).to(device)
    tokenizer = model.tokenizer
    d = model.get_sentence_embedding_dimension()
    print(f"    ⇒ embedding dim = {d}")

    docs, dids = [], []
    with open(clean_document_path) as f:
        for line in f:
            obj = json.loads(line)
            dids.append(obj["did"])
            docs.append(obj["cleaned"])

    var_dict = {}

    for i in tqdm(range(0, len(docs), batch_size), desc="MC-dropout (Fully Optimized)"):
        batch_texts = docs[i:i+batch_size]
        batch_dids  = dids[i:i+batch_size]

        var_scores = mc_dropout_variance_optimized(batch_texts,
                                                   model, tokenizer,
                                                   T=T, device=device)

        for did, v in zip(batch_dids, var_scores.tolist()):
            var_dict[did] = v

    with open(variance_dict_file, "w") as f:
        json.dump(var_dict, f)
    print(f"[✓] saved variance dictionary to {variance_dict_file}")


if __name__ == "__main__":
    # ... (argparse 부분은 동일) ...
    ap = argparse.ArgumentParser("MC-Dropout variance for ST models")
    ap.add_argument("--base_modelname_or_path", required=True, type=str)
    ap.add_argument("--clean_document_path",    required=True, type=str)
    ap.add_argument("--variance_dict_file",     required=True, type=str)
    ap.add_argument("--cache_dir",              required=True, type=str)
    ap.add_argument("--T",          type=int, default=10, help="# Monte-Carlo samples per document")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device",     type=str, default="cuda")
    args = ap.parse_args()
    build_variance_dict(**vars(args))