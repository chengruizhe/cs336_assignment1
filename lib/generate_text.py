import torch

from cs336_basics.transformer import Transformer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.softmax import Softmax


def generate_text(
    model: Transformer,
    tokenizer: Tokenizer,
    input_text: str,
    max_tokens: int,
    softmax_temp: float,
    top_p: float,
) -> str:
    assert softmax_temp > 0.0, "softmax_temp must be positive"
    assert 0.0 < top_p <= 1.0, "top_p must be in the range (0, 1]"
    
    vocab_size = tokenizer.vocab_size
    max_seq_len = model.context_length

    tokens = tokenizer.encode(input_text)
    model.eval()
    device = next(model.parameters()).device
    softmax = Softmax()

    end_token: int = tokenizer.encode("<|endoftext|>")[0]
    next_token: int = -1
    num_new_tokens = 0

    with torch.inference_mode():
        while num_new_tokens < max_tokens and next_token != end_token:
            cur_window_tokens = tokens[-max_seq_len:]
            input_ids = torch.tensor(
                [cur_window_tokens], device=device, dtype=torch.long
            )
            logits = model(input_ids)
            assert logits.shape == (
                1,
                len(cur_window_tokens),
                vocab_size,
            )

            next_token_logits = logits[0, -1, :] / softmax_temp
            next_token_probs = softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)

            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_remove_mask = cumulative_probs > top_p
            sorted_remove_mask[1:] = sorted_remove_mask[:-1].clone()
            sorted_remove_mask[0] = False

            sorted_indices_to_remove = sorted_indices[sorted_remove_mask]
            next_token_logits[sorted_indices_to_remove] = float("-inf")
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(
                next_token_probs,
                num_samples=1,
                replacement=False,
            ).item()

            tokens.append(next_token)
            num_new_tokens += 1

    return tokenizer.decode(tokens)
