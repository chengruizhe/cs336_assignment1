import pathlib
import json
from tests.adapters import run_train_bpe
from tests.common import gpt2_bytes_to_unicode


if __name__ == "__main__":
    root_path = pathlib.Path(__file__).parent.parent
    # input_path = root_path / "data/TinyStoriesV2-GPT4-train.txt"
    # output_path = root_path / "models/TinyStoriesV2-train/"
    input_path = root_path / "data/owt_train.txt"
    output_path = root_path / "models/owt_train/"
    output_path.mkdir(parents=True, exist_ok=True)
    
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    
    unicode_map = gpt2_bytes_to_unicode()
    with open(output_path / "vocab.json", "w") as f:
        json.dump(
            {"".join(unicode_map[b] for b in v): k for k, v in vocab.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
            
    with open(output_path / "merges.txt", "w") as merges_file:
        for merge in merges:
            m1 = "".join(unicode_map[b] for b in merge[0])
            m2 = "".join(unicode_map[b] for b in merge[1])
            merges_file.write(m1)
            merges_file.write(" ")
            merges_file.write(m2)
            merges_file.write("\n")