import torch
from torch.nn import Module
from typing import Union, List
from harp import HARPWrapper
from .logits_processor import LogitsProcessor, GreedyProcessor
from typing import List


@torch.no_grad()
def autoregressive_generate(
    inputs: List[int],
    model: Union[Module, HARPWrapper],
    max_gen_len: int = 120,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: Union[int, List[int]] = 1,
    pad_token_id: int = 0,
    vanilla: bool = False,
) -> List[int]:
    """
    Generate text sequence autoregressively based on the input sequence.

    Args:
        inputs (List[int]): input token ids.
        model (Module, HARPWrapper): model to use for inference.
        max_gen_len (int): maximum length of the generated sequence. Default is 120.
        logits_processor (LogitsProcessor): logits processor for predicting the tokens. Default is GreedyProcessor.
        eos_token_id (int | List[int]): end token id. Default is 1.
        pad_token_id (int): pad token id. Default is 0.
        vanilla (bool): whether to use vanilla model or hesitation model (applies only if the model is of type HARPWrapper).

    Returns:
        List[int]: generated sequence token ids.

    Note:
        This generation methods only works for decoder-only models.
        The cache is not used in this method.
        Only works for batch size 1.
    """
    is_harp_wrapped = isinstance(model, HARPWrapper)

    prompt_len = len(inputs)
    max_seq_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else (model.config.max_context_length if hasattr(model.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)

    if not vanilla:
        assert is_harp_wrapped, "Please wrap your model into HARPWrapper to use Hesitation-Aware generation."
        
    
    start = prompt_len
    for curr in range(start, total_len):
        if vanilla and not is_harp_wrapped:
            o = model(input_ids[..., :curr])
        else:
            o = model(
                input_ids=input_ids[..., :curr],
                vanilla=vanilla,
            )
        logits = o.logits[..., -1, :]  # (1, vocab_size)
        distribution = logits_processor(logits)
        x = logits_processor.sample(distribution)  # (1, 1)
        input_ids[0, curr] = x

        # check for end token
        if torch.isin(x, stop_tokens):
            curr -= 1
            break
        
    return input_ids[0, prompt_len : curr + 1].tolist()