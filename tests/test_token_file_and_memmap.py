from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lib.memmap_dataset import MemmapTokenDataset
from scripts.run_tokenizer import encode_and_save_tokens


class _TinyTokenizer:
    def __init__(self, special_token: str = "<|endoftext|>", special_id: int = 60000) -> None:
        self.special_token = special_token
        self.special_id = special_id

    def encode(self, text: str) -> list[int]:
        if text == self.special_token:
            return [self.special_id]

        result: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith(self.special_token, i):
                result.append(self.special_id)
                i += len(self.special_token)
            else:
                result.append(ord(text[i]))
                i += 1
        return result


def test_encode_and_save_tokens_inserts_boundary_token_across_chunks(monkeypatch, tmp_path: Path) -> None:
    text = "abcdefghij"
    input_path = tmp_path / "sample.txt"
    output_path = tmp_path / "sample.bin"
    input_path.write_text(text)

    monkeypatch.setattr("scripts.run_tokenizer.find_chunk_boundaries", lambda **_: [0, 5, len(text)])
    tokenizer = _TinyTokenizer()

    encode_and_save_tokens(
        tokenizer=tokenizer,
        input_path=input_path,
        output_path=output_path,
        text_chunk_size=4,
    )

    tokens = np.fromfile(output_path, dtype=np.uint16).tolist()
    expected = [ord(c) for c in "abcde"] + [tokenizer.special_id] + [ord(c) for c in "fghij"]
    assert tokens == expected


def test_generated_tokens_load_with_memmap_dataset(monkeypatch, tmp_path: Path) -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    input_path = tmp_path / "tiny.txt"
    output_path = tmp_path / "tiny.bin"
    input_path.write_text(text)

    monkeypatch.setattr("scripts.run_tokenizer.find_chunk_boundaries", lambda **_: [0, 20, len(text)])
    tokenizer = _TinyTokenizer()
    encode_and_save_tokens(
        tokenizer=tokenizer,
        input_path=input_path,
        output_path=output_path,
        text_chunk_size=8,
    )

    arr = np.memmap(output_path, mode="r", dtype=np.uint16)
    dataset = MemmapTokenDataset(arr)
    x, y = dataset.sample_batch(batch_size=4, context_length=5, device=torch.device("cpu"))
    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
