import pathlib
import regex
import json
from typing import Iterable
from ..tests.common import gpt2_bytes_to_unicode

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.index_to_token = vocab
        self.token_to_index = {v: k for k, v in vocab.items()}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(regex.escape(t) for t in special_tokens) + ")"
        else:
            self.special_pattern = None
            
    @property
    def vocab_size(self) -> int:
        return len(self.index_to_token)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | pathlib.Path,
        merges_path: str | pathlib.Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        gpt2_bytes_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {
                int(index): bytes(gpt2_bytes_decoder[t] for t in item) for item, index in vocab.items()
            }
            
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes(gpt2_bytes_decoder[token] for token in merge_token_1),
                    bytes(gpt2_bytes_decoder[token] for token in merge_token_2),
                )
                for merge_token_1, merge_token_2 in merges
            ]
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )
        
    def encode(
        self,
        text: str,
    ) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        tokens = []
        if self.special_pattern is not None:
            chunks = regex.split(self.special_pattern, text)
        else:
            chunks = [text]
        for subchunk in chunks:
            if not subchunk:
                continue
            if subchunk in self.special_tokens:
                tokens.append(self.token_to_index[subchunk.encode("utf-8")])
                continue
            for match in regex.finditer(PAT, subchunk):
                token = match.group(0).encode("utf-8")
                token_bytes = [bytes([b]) for b in token]
                
                while True:
                    pairs = [(token_bytes[i], token_bytes[i + 1]) for i in range(len(token_bytes) - 1)]
                    candidates = [
                        (idx, self.merges[pair]) for idx, pair in enumerate(pairs) if pair in self.merges
                    ]
                    if not candidates:
                        break
                    min_idx, _ = min(candidates, key=lambda x: x[1])
                    token_bytes[min_idx: min_idx + 2] = [token_bytes[min_idx] + token_bytes[min_idx + 1]]
                for t in token_bytes:
                    tokens.append(self.token_to_index[t])
        return tokens
    
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterable[int]:
        for word in iterable:
            yield from self.encode(word)
    
    def decode(
        self,
        ids: list[int],
    ) -> str:
        result = b"".join(self.index_to_token[i] for i in ids)
        return result.decode("utf-8", errors="replace")