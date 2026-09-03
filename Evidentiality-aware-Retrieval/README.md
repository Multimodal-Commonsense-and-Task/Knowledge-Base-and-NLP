# Evidentiality-Aware Dense Passage Retrieval (EADPR)

[*Evidentiality-aware Retrieval for Overcoming Abstractiveness in Open-Domain Question Answering*](https://aclanthology.org/2024.findings-eacl.130/) (Findings of EACL 2024) 의 재현 구현.

Yongho Song\*, Dahyun Lee\*, Myungha Jang, Seung-won Hwang, Kyungjae Lee, Dongha Lee, Jinyoung Yeo

> ⚠ 이 저장소는 **논문의 방법론을 코드로 옮긴 것**이지 저자 공식 구현이 아니다.
> 논문의 수치를 재현하지 않는다 — 아래 [Scope](#scope) 참고.

## Overview

abstractive ODQA 에서 **relevance 와 answerability 는 어긋난다.** 검색기는 질문과
의미적으로 가까운 패시지를 잘 찾지만, 그 패시지가 답의 근거를 담고 있다는 보장은 없다.
표준 IR 데이터셋은 relevance 주석만 주므로 evidentiality 에 대한 지도 신호가 약하다.

기존 해법은 reader 신호로 answerability 를 반복 주석하는 **model-centric** 접근인데
연산 비용이 크다. EADPR 은 대신 **data-centric** 으로 간다 — gold evidence passage 에서
근거 span 을 잘라내 **distractor** 를 합성하고, 이 distractor 를 학습 신호로 쓴다.

핵심 착상은 distractor 를 **양성과 음성 사이의 pivot** 으로 놓는 것이다.
distractor 는 여전히 질문과 관련은 있지만 근거는 없다:

```
        <q, p+>   >   <q, p*>   >   <q, p->
        evidence      distractor    irrelevant
                 └ Eq.2            └ Eq.4
```

이 두 부등식을 각각 손실로 만들어 (distractor 를 hard negative 로, 동시에
pseudo-positive 로) 표준 DPR 목적함수 위에 얹는다.

## Method

### §3.1 Distractor 증강 — `src/distractors.py`

gold passage `p+ = [s_l ; s+ ; s_r]` 에서 근거 span `s+` 를 빼면 `p* = [s_l ; s_r]` 이 된다.
문제는 대부분의 데이터셋에 evidence span 주석이 없다는 것. 논문은 **pseudo-evidence** 로 우회한다:

1. `p+` 를 n 개 span 으로 나누고, 하나씩 빼서 후보 `{p*_i}` 를 만든다.
2. 생성형 QA 모델 θ (논문은 UnifiedQA-T5) 에 `(q, p*_i)` 를 넣어 정답의 confidence
   `P_θ(a | q, p*_i)` 를 잰다.
3. **confidence 가 가장 낮은 = perplexity 가 가장 높은** 후보를 `p*` 로 고른다.
   confidence 가 급락했다는 건 그 span 이 답에 기여했다는 뜻이다.

### §3.2 Evidentiality-aware learning — `src/losses.py`

세 손실의 가중합 (Eq.8):

```
L_eadpr = L_dpr + τ1·L_HN + τ2·L_PP
```

| 손실 | 수식 | 역할 |
|---|---|---|
| `L_HN` (Eq.3) | `-log( e^<q,p+> / (e^<q,p+> + e^<q,p*>) )` | distractor 를 **hard negative** 로. p+ 를 p* 위로 민다 |
| `L_PP` (Eq.5) | `-log( e^<q,p*ᵢ> / (e^<q,p*ᵢ> + Σ_{j≠i}(e^<q,p⁻ⱼ> + e^<q,p*ⱼ>)) )` | distractor 를 **pseudo-positive** 로. 무관한 패시지 위로 올린다 |
| `L_dpr` (Eq.7) | 표준 DPR 분모에 `λ·e^<q,p*ᵢ>` 추가 | 자기 distractor 를 λ 가중 음성으로 |

`λ=0` 이면 `L_dpr` 은 원래 DPR (Eq.1) 로 정확히 환원된다 — 테스트로 확인한다.

### §5 Answer-Awareness — `src/evaluate.py`

정답 span 을 지운 `p'` 를 만들어, 모델이 `p+` 를 `p'` 보다 높게 매기는 비율을 잰다 (Eq.9):

```
AA = 1 - (1/T) Σ 1[ <q,p+> ≤ <q,p'> ]
```

논문의 관찰은 vanilla DPR 의 AA 가 이론적 상한에 크게 못 미치고, 특히 답이 named entity 인
who/what/where 질문에서 낮다는 것이다. 구현은 질문 유형별로도 쪼개서 낸다.

## Repository Structure

```
.
├── main.py                # CLI — toy / augment / train / retrieve / aa / robustness
├── requirements.txt
├── tests/test_losses.py   # 손실이 논문 수식과 맞는지 검증
└── src/
    ├── config.py          # Appendix C 하이퍼파라미터
    ├── data.py            # 스키마 · JSONL io · span 분할 · 토이 데이터 생성기
    ├── distractors.py     # §3.1 span 제거 + QA perplexity 로 pseudo-evidence 선택
    ├── modeling.py        # DPR dual encoder [f_q, f_p] + 오프라인 폴백
    ├── losses.py          # Eq.3 / Eq.5 / Eq.7 / Eq.8
    ├── train.py           # 학습 루프 (in-batch negative)
    ├── retrieval.py       # 코퍼스 인코딩 · 검색 · Top-k / MRR / R@k
    └── evaluate.py        # AA score (Eq.9) · robustness 시뮬레이션
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. 합성 데모 데이터 (다운로드 불필요)
python main.py toy

# 2. §3.1 distractor 증강
python main.py augment --tiny --split train
python main.py augment --tiny --split dev

# 3. §3.2 EADPR 학습
python main.py train --tiny --epochs 5

# 4. 평가
python main.py retrieve   --tiny        # Top-k accuracy · MRR · R@k
python main.py aa         --tiny        # Answer-Awareness (Eq.9)
python main.py robustness --tiny        # 코퍼스에 distractor 주입

# 손실 수식 검증
python tests/test_losses.py
```

`--tiny` 는 사전학습 가중치를 받지 않고 **소형 랜덤 초기화 모델**로 돌린다 —
파이프라인이 끝까지 도는지 확인하기 위한 경로이며 성능은 의미가 없다.
실제 설정은 `--tiny` 를 빼면 된다 (BERT-base + UnifiedQA-T5 를 내려받는다):

```bash
python main.py augment --split train --device cuda
python main.py train   --device cuda --epochs 40
```

### 데이터 포맷

자체 데이터를 쓰려면 `--data-dir` 아래에 JSONL 세 개를 두면 된다.

`train.jsonl` / `dev.jsonl` — 한 줄이 한 질문:

```json
{"qid": "q-0", "question": "...", "answers": ["..."],
 "positive_ctx": {"pid": "...", "title": "...", "text": "..."},
 "hard_negative_ctxs": [{"pid": "...", "title": "...", "text": "..."}],
 "supporting_pids": ["..."]}
```

`corpus.jsonl` — `{"pid", "title", "text"}` 한 줄씩.
`distractor_ctx` 는 `augment` 가 채운다. `supporting_pids` 는 multi-hop R@k 에만 쓰인다.

### 주요 옵션

| 옵션 | 기본값 | 대응 |
|---|---|---|
| `--lambda-distractor` | 1.0 | Eq.7 의 λ. `{0.1,0.2,0.5,0.9,1.0}` 중 논문은 1.0 |
| `--tau-hn` / `--tau-pp` | 1.0 / 1.0 | Eq.8 의 τ1 / τ2. grid search 로 결정 |
| `--no-hn` / `--no-pp` | — | 해당 항 제거 (ablation) |
| `--no-qa-model` | — | QA 모델 없이 길이 휴리스틱으로 distractor 선택 |
| `--threads` | 4 | CPU 스레드 상한. **아래 주의 참고** |

하이퍼파라미터 기본값은 논문 Appendix C 를 따른다 — BERT-base, 40 epochs,
batch 16, lr 2e-5, Adam eps 1e-8 / betas (0.9, 0.999).

> **`--threads` 주의.** 코어가 많은 노드에서 작은 텐서를 돌리면 스레드 오버헤드가
> 연산을 압도한다. 이 저장소의 tiny 경로에서 52 스레드 대비 4 스레드가 **100 배 이상**
> 빨랐다 (배치당 14s → 0.1s 미만). 기본값 4 를 두고 쓰다가, 큰 모델로 갈 때만 올릴 것.

## Scope

**구현한 것** — §3.1 distractor 증강 (span 제거 + QA perplexity 선택), §3.2 세 손실 전부,
DPR dual encoder 학습 루프, Top-k / MRR / R@k, AA score, distractor 주입 robustness 실험,
`--no-hn` / `--no-pp` ablation.

**구현하지 않은 것** — 아래는 논문에 있으나 여기 없다:

- **reader.** DPR extractive reader · FiD · ELECTRA 를 붙인 end-to-end QA (Table 3/5) 는 없다.
  검색기까지만 다룬다.
- **negative mining.** BM25 · ANCE 로 hard negative 를 캐는 파이프라인은 없다.
  `hard_negative_ctxs` 필드로 **받아 쓰기만** 한다.
- **MDR.** multi-hop dense retrieval (Table 4/5) 은 없다. `supporting_pids` 기반 R@k 만 낸다.
- **실제 벤치마크.** NQ / TriviaQA / TREC / HotpotQA 와 21M Wikipedia 코퍼스는 포함하지 않는다.
  동봉된 것은 합성 토이 데이터뿐이다.
- **FAISS 인덱스.** 검색은 전수 내적이다. 토이 규모에서는 충분하지만 대형 코퍼스에는 부족하다.

즉 **논문 수치를 재현하지 않는다.** 논리 구조와 실행 가능성까지가 이 저장소의 범위다.

## Correctness

성능이 목표가 아니므로, 대신 **수식과 코드가 일치하는지**를 테스트로 고정했다
(`tests/test_losses.py`, 7 케이스 통과):

- 순서가 옳은 배치(`p+ > p* > p-`)가 뒤집힌 배치보다 손실이 낮다
- `L_HN` 이 Eq.3 의 이항 softmax 와 수치적으로 일치한다
- `λ=0` 일 때 `L_dpr` 이 표준 DPR (Eq.1) 로 정확히 환원된다
- λ 를 키우면 `L_dpr` 이 단조 증가한다
- 세 손실 모두로 그래디언트가 흐른다 (유한하고 0 이 아니다)
- hard negative 를 넣으면 `L_PP` 분모가 커져 손실이 증가한다
- `--no-hn --no-pp` 가 `L_eadpr` 을 `L_dpr` 로 정확히 되돌린다

학습 중에는 두 부등식이 실제로 지켜지는 비율을 `acc_pos_over_dis` (Eq.2) 와
`acc_dis_over_neg` (Eq.4) 로 매 epoch 찍는다.

## Citation

```bibtex
@inproceedings{song-etal-2024-evidentiality,
    title = "Evidentiality-aware Retrieval for Overcoming Abstractiveness in Open-Domain Question Answering",
    author = "Song, Yongho and Lee, Dahyun and Jang, Myungha and Hwang, Seung-won and
              Lee, Kyungjae and Lee, Dongha and Yeo, Jinyoung",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    year = "2024",
    pages = "1930--1943",
}
```
