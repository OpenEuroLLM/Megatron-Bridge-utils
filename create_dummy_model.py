#!/usr/bin/env python3

# Save a dummy HF model based on config and tokenizer to use as
# reference for conversion with bridge. This is part of a workaround
# for bridge.export_ckpt() for some reason requiring a full model,
# i.e. initialization via AutoBridge.from_hf_pretrained() rather than
# AutoBridge.from_hf_config().

import sys
import os

from argparse import ArgumentParser

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('tokenizer')
    ap.add_argument('outdir')
    return ap.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.outdir) and os.listdir(args.outdir):
        print(f'{args.outdir} is not empty, not clobbering', file=sys.stderr)
        return 1

    c = AutoConfig.from_pretrained(args.config)
    t = AutoTokenizer.from_pretrained(args.tokenizer)
    m = AutoModelForCausalLM.from_config(c)

    m.save_pretrained(args.outdir)
    c.save_pretrained(args.outdir)
    t.save_pretrained(args.outdir)


if __name__ == '__main__':
    sys.exit(main())
