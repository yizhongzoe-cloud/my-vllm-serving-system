"""Convert RULER prepare.py output to our jsonl format.

RULER format: {"index", "input", "outputs", "length", ...}
Our format:   {"id", "dataset", "prompt", "prompt_tokens", "expected_output_tokens", "subtask"}

Usage:
    python convert_ruler.py \
        --src /tmp/ruler_out/niah_single_1/validation.jsonl \
        --dst experiments_v2/datasets/cached/ruler_64k_niah.jsonl \
        --dataset-name ruler_64k_niah \
        --subtask niah_single_1 \
        --expected-output-tokens 128
"""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--subtask", type=str, required=True)
    parser.add_argument("--expected-output-tokens", type=int, default=128)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.src) as fin, open(args.dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = d["input"]
            prompt_tokens = len(tok.encode(prompt))
            rec = {
                "id": f"{args.subtask}_{d['index']}",
                "dataset": args.dataset_name,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "expected_output_tokens": args.expected_output_tokens,
                "subtask": args.subtask,
            }
            fout.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {args.dst}")


if __name__ == "__main__":
    main()
