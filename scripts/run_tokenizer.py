import pathlib
from tqdm import tqdm
import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


def get_tiny_stories_tokenizer(root_path: pathlib.Path) -> Tokenizer:
    vocab_path = root_path / "models/TinyStoriesV2-train/vocab.json"
    merges_path = root_path / "models/TinyStoriesV2-train/merges.txt"
    return Tokenizer.from_files(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=["<|endoftext|>"],
    )


def get_owt_tokenizer(root_path: pathlib.Path) -> Tokenizer:
    vocab_path = root_path / "models/owt_train/vocab.json"
    merges_path = root_path / "models/owt_train/merges.txt"
    return Tokenizer.from_files(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=["<|endoftext|>"],
    )


def encode_and_save_tokens(
    tokenizer: Tokenizer,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    text_chunk_size: int = 8000000,
    boundary_special_token: str = "<|endoftext|>",
) -> None:
    file_size = input_path.stat().st_size
    boundary_token_ids = tokenizer.encode(boundary_special_token)
    assert len(boundary_token_ids) == 1, (
        f"Expected exactly one token id for boundary special token `{boundary_special_token}` "
        f"but got {boundary_token_ids}"
    )
    boundary_token_id = boundary_token_ids[0]

    with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Encoding") as pbar:
        with (
            open(input_path, "rb") as f,
            open(output_path, "wb") as out,
        ):
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=(file_size // text_chunk_size) + 1,
                split_special_token=boundary_special_token.encode("utf-8"),
            )

            for chunk_idx, (start, end) in enumerate(
                zip(chunk_boundaries[:-1], chunk_boundaries[1:])
            ):
                f.seek(start)
                text_chunk = f.read(end - start).decode("utf-8")
                tokens = tokenizer.encode(text_chunk)
                if chunk_idx > 0 and (not tokens or tokens[0] != boundary_token_id):
                    tokens = [boundary_token_id] + tokens
                assert max(tokens) < 2**16, "Token ID exceeds uint16 limit"

                array = np.array(tokens, dtype=np.uint16)
                array.tofile(out)
                pbar.update(end - start)
    print(
        f"Done! Output saved to: {output_path}. "
        f"Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB"
    )


if "__main__" == __name__:
    root_path = pathlib.Path(__file__).parent.parent
    tiny_stories_tokenizer = get_tiny_stories_tokenizer(root_path)
    owt_tokenizer = get_owt_tokenizer(root_path)

    print(f"TinyStoriesTokenizer vocab size: {tiny_stories_tokenizer.vocab_size}")
    print(f"OwtTokenizer vocab size: {owt_tokenizer.vocab_size}")

    input_paths = {
        "TinyStories-valid": root_path / "data/TinyStoriesV2-GPT4-valid.txt",
        "owt-valid": root_path / "data/owt_valid.txt",
        "TinyStories-train": root_path / "data/TinyStoriesV2-GPT4-train.txt",
        "owt-train": root_path / "data/owt_train.txt",
    }

    encode_and_save_tokens(
        tokenizer=tiny_stories_tokenizer,
        input_path=input_paths["TinyStories-train"],
        output_path=root_path / "data/TinyStoriesV2-GPT4-train-tokens.bin",
    )
    encode_and_save_tokens(
        tokenizer=tiny_stories_tokenizer,
        input_path=input_paths["TinyStories-valid"],
        output_path=root_path / "data/TinyStoriesV2-GPT4-valid-tokens.bin",
    )
    encode_and_save_tokens(
        tokenizer=tiny_stories_tokenizer,
        input_path=input_paths["owt-train"],
        output_path=root_path / "data/owt-train-tokens.bin",
    )
    encode_and_save_tokens(
        tokenizer=tiny_stories_tokenizer,
        input_path=input_paths["owt-valid"],
        output_path=root_path / "data/owt-valid-tokens.bin",
    )

    # for name, input_path in input_paths.items():
    #     with open(input_path, "r") as f:
    #         text = f.read()
    #         for tokenizer_name, tokenizer in {
    #             "TinyStories": tiny_stories_tokenizer,
    #             "Owt": owt_tokenizer,
    #         }.items():
    #             tokens = tokenizer.encode(text)
    #             compression_ratio = len(text) / len(tokens)
    #             print(
    #                 f"Compression ratio for {name} with {tokenizer_name}: {compression_ratio:.2f}"
    #             )
