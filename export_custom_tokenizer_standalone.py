#!/usr/bin/env python3
"""
Patch a HuggingFace export directory to use a custom tokenizer.

Replaces the tokenizer files written by convert_checkpoints.py and updates
config.json so that vocab_size and special token IDs match the custom tokenizer.

Usage:
  python export_custom_tokenizer_standalone.py \\
      --hf-path ./output \\
      --tokenizer-path openai/gpt-oss-120b
"""

import argparse
import json
import sys
from pathlib import Path


def patch_config_and_tokenizer(hf_path: Path, tokenizer_path: str) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required: pip install transformers") from exc

    print(f"Loading custom tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"  vocab size  : {len(tokenizer)}")
    print(f"  type        : {type(tokenizer).__name__}")
    print(f"  bos_token_id: {tokenizer.bos_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")

    print(f"\nSaving custom tokenizer to: {hf_path}")
    tokenizer.save_pretrained(hf_path)
    tokenizer_files = (
        list(hf_path.glob("tokenizer*"))
        + list(hf_path.glob("vocab*"))
        + list(hf_path.glob("merges.txt"))
    )
    for tf in sorted(tokenizer_files):
        print(f"  wrote: {tf.name}")

    config_file = hf_path / "config.json"
    if not config_file.exists():
        print(f"WARNING: {config_file} not found — skipping config patch")
        return

    with config_file.open() as f:
        config = json.load(f)

    changed: list[str] = []

    custom_vocab_size = len(tokenizer)
    if config.get("vocab_size") != custom_vocab_size:
        changed.append(f"  vocab_size: {config.get('vocab_size')} -> {custom_vocab_size}")
        config["vocab_size"] = custom_vocab_size

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        tok_val = getattr(tokenizer, attr, None)
        if tok_val is not None and config.get(attr) != tok_val:
            changed.append(f"  {attr}: {config.get(attr)} -> {tok_val}")
            config[attr] = tok_val

    if changed:
        print("\nPatching config.json:")
        for line in changed:
            print(line)
        with config_file.open("w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"  saved: {config_file}")
    else:
        print("\nconfig.json already matches custom tokenizer — no changes needed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch a HuggingFace export to use a custom tokenizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="Directory containing the exported HuggingFace model",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="HuggingFace tokenizer ID or local path (e.g. 'openai/gpt-oss-120b')",
    )

    args = parser.parse_args()
    hf_path = Path(args.hf_path)

    if not hf_path.exists():
        print(f"ERROR: hf-path does not exist: {hf_path}", file=sys.stderr)
        return 1

    patch_config_and_tokenizer(hf_path=hf_path, tokenizer_path=args.tokenizer_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
