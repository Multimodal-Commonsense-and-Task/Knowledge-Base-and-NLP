import torch
from transformers import AutoModelForCausalLM
from typing import Optional


def drop_out(embeddings: torch.Tensor, delta: float):
    """
    Drop out embeddings with the given rate.

    Args:
        embeddings (torch.Tensor): The embeddings to apply drop out to. (batch_size, sequence_length, hidden_size).
        delta (float): The rate of drop out.

    Returns:
        torch.Tensor: The embeddings with drop out applied. (batch_size, sequence_length, hidden_size).
    """
    mask = torch.rand_like(embeddings) > delta
    return embeddings * mask


def shannon_entropy(logits: torch.Tensor):
    """
    Returns the Shannon entropy of the model last-position logits.

    Args:
        logits (Tensor): The model logits. (batch_size, seq_len, vocab_size)
        
    Returns:
        torch.Tensor[float]: The Shannon entropy of the model last-position logits. (batch_size)
    """
    logits = logits[..., -1, :]
    distributions = torch.softmax(logits, dim=-1)
    return -torch.sum(distributions * torch.log2(distributions + 1e-15), dim=-1)


class HesitationOutput:
    def __init__(self, logits: torch.Tensor, hesitations: Optional[torch.Tensor] = None):
        """
        Represents the output of the Hesitation model.
        
        Args:
            logits (torch.Tensor): The logits from the model.
            hesitations (Optional[torch.Tensor]): The hesitations for each batch.
        """
        self.logits = logits
        self.hesitations = hesitations
        
        
class HARPWrapper(torch.nn.Module):
    
    def __init__(self, model: AutoModelForCausalLM, theta: Optional[float] = 1.0, delta: Optional[float] = 0.2, beta: Optional[float] = 0.5):
        """
        Wrapper class for the Hesitation-Aware Reframing Forward Pass (HARP).
        
        Args:
            model (AutoModelForCausalLM): The underlying transformers model.
            theta (float): The threshold of uncertainty. Default is 1.0.
            delta (float): The dropout rate for the reframed embeddings. Default is 0.2.
            beta (float): The weight for the reframed logits. Default is 0.5.
        """
        super().__init__()
        self.model = model
        self.embeddings = model.get_input_embeddings()
        self.config = model.config
        self.device = model.device
        
        # Hyperparameters
        self.theta = theta
        self.delta = delta
        self.beta = beta
        
        self.model.eval()  # HARP is only for inference

    def forward(
        self,
        input_ids: torch.Tensor,
        vanilla: bool = False,
        **kwargs
    ) -> HesitationOutput:
        """
        Forward pass of the HARP model.
        
        Args:
            input_ids (torch.Tensor): The input IDs.
            vanilla (bool): Whether to use the model as vanilla or hesitation-aware. Default is False (Hesitation-aware).
            **kwargs: Additional keyword arguments for the underlying model.
            
        Returns:
            HesitationOutput: The output of the HARP model.
        """
        input_embeddings = self.embeddings(input_ids)
        logits = self.model(inputs_embeds=input_embeddings, use_cache=False, **kwargs).logits  # logits from the vanilla model
        
        if vanilla:
            return HesitationOutput(logits)
        
        hesitation = shannon_entropy(logits)  # compute the hesitation for each batch
        mask = (hesitation > self.theta)  # boolean mask for each batch, True if the model is uncertain for the given batch
        
        if not mask.any():
            return HesitationOutput(logits, hesitations=hesitation)
        
        reframed_embeddings = drop_out(input_embeddings, self.delta)  # reframe the input embeddings
        logits_r = self.model(inputs_embeds=reframed_embeddings, use_cache=False, **kwargs).logits  # reframed logits after processing the reframed embeddings
        logits[mask, -1, :] = self.beta * logits[mask, -1, :] + (1 - self.beta) * logits_r[mask, -1, :]  # combine the vanilla logits and the reframed logits
        
        return HesitationOutput(logits, hesitations=hesitation)
