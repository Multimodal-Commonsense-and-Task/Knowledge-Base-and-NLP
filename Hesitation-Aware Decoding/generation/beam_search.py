import torch
from torch.nn import Module
from typing import Union, List
from harp import HARPWrapper
from typing import List


@torch.no_grad()
def beam_search_generate(
    inputs: List[int],
    model: Union[Module, HARPWrapper],
    max_gen_len: int = 120,
    num_beams: int = 3,
    top_k: int = 5,
    min_length: float = 5.0,
    alpha: float = 1.2,
    eos_tokens_id: Union[int, List[int]] = 1,
    pad_token_id: int = 0,
) -> List[int]:
    """
    Generate text sequence using beam search based on the input sequence.

    Args:
        inputs (List[int]): input token ids of batch size 1.
        model (Module): model to use for inference.
        max_gen_len (int): maximum length of the generated sequence. Default is 120.
        num_beams (int): number of beams. Default is 3.
        top_k (int): number of top k to consider at each beam. Default is 5.
        min_length (float): length penalty. Default is 5.0.
        alpha (float): alpha parameter of beam search decoding. Default is 1.2.
        eos_token_id (int | List[int]): end token id. Default is 1.
        pad_token_id (int): pad token id. Default is 0.

    Returns:
        List[int]: generated sequence token ids.

    Note:
        This generation methods only works for decoder-only models.
        The cache is not used in this method.
        Only works for batch size 1.
        Beam Search is only adapted for Greedy decoding.
    """

    def _length_penalty_fn(length, alpha, min_length):
        return ((min_length + length) / (min_length + 1)) ** alpha

    is_harp_wrapper = isinstance(model, HARPWrapper)
    prompt_len = len(inputs)
    max_seq_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else (model.config.max_context_length if hasattr(model.config, 'max_context_length') else 1024)

    assert prompt_len < max_seq_length, "Prompt length exceeds maximum sequence length."

    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((num_beams, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[:, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)
    probs = torch.full((num_beams, total_len), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    beams_probs = torch.full((num_beams,), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    last_indexes = torch.full((num_beams,), -1, dtype=torch.long, device=model.device)

    stop_tokens = torch.tensor((eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]), dtype=torch.long, device=model.device)

    # prefill
    probs[:, :prompt_len] = 1.0
    beams_probs[:] = 1.0
    o = model(input_ids[:, :prompt_len], vanilla=True) if is_harp_wrapper else model(input_ids[:, :prompt_len])
    curr_prob = torch.nn.functional.log_softmax(o.logits[0, -1, :], dim=-1)
    top_probs, top_tokens = torch.topk(curr_prob, num_beams, dim=-1)
    input_ids[:, prompt_len] = top_tokens
    probs[:, prompt_len] = probs[:, prompt_len - 1] + top_probs
    beams_probs[:] = probs[:, prompt_len] / _length_penalty_fn(1, alpha, min_length)
    
    for curr in range(prompt_len + 1, total_len):
        if is_harp_wrapper:
            o = model(input_ids=input_ids[:, :curr], vanilla=True)
        else:
            o = model(input_ids[:, :curr])
        logits = o.logits[:, -1, :]
        probs_curr = torch.nn.functional.log_softmax(logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs_curr, top_k, dim=-1)
        possibilities = []
        for beam in range(num_beams):
            if last_indexes[beam] != -1:
                prob_vec = probs[beam].detach().clone()
                input_vec = input_ids[beam].detach().clone()
                possibilities.append(
                    (beams_probs[beam], input_vec, prob_vec, last_indexes[beam])
                )
                continue
            
            for possibility in range(top_k):
                new_prob = probs[beam, curr - 1] + top_probs[beam, possibility]
                lp = _length_penalty_fn(curr - prompt_len, alpha, min_length)
                prob_vec = probs[beam].detach().clone()
                prob_vec[curr] = new_prob
                input_vec = input_ids[beam].detach().clone()
                input_vec[curr] = top_tokens[beam, possibility]
                last_token_idx = -1
                if torch.isin(input_vec[curr], stop_tokens) or input_vec[curr] == pad_token_id:
                    last_token_idx = curr
                    
                already_in = False
                for p in possibilities:
                    if torch.equal(p[1], input_vec):
                        already_in = True
                        break
                if not already_in:
                    possibilities.append(
                        (new_prob / (lp if lp != 0 else 1), input_vec, prob_vec, last_token_idx)
                    )

        possibilities.sort(key=lambda x: x[0], reverse=True)
        possibilities = possibilities[:num_beams]

        for beam in range(num_beams):
            beams_probs[beam] = possibilities[beam][0]
            input_ids[beam] = possibilities[beam][1]
            probs[beam] = possibilities[beam][2]
            last_indexes[beam] = possibilities[beam][3]

        if torch.all(last_indexes != -1):
            break

    last_indexes[last_indexes == -1] = total_len - 1

    return input_ids[0, prompt_len : last_indexes[0] + 1].tolist()
