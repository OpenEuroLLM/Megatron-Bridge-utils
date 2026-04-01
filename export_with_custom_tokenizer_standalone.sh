#!/bin/bash
# Export a Megatron checkpoint to HuggingFace format with a custom tokenizer.
#
# Usage:
#   ./export_with_custom_tokenizer_standalone.sh \
#       <megatron_path> <hf_path> <hf_model> <tokenizer_path> \
#       [path/to/Megatron-Bridge] [path/to/patch_script.py]
#
# Arguments:
#   megatron_path   - Directory containing the Megatron checkpoint
#   hf_path         - Output directory for the HuggingFace model
#   hf_model        - HF model ID for architecture reference (e.g. Qwen/Qwen3-0.6B)
#   tokenizer_path  - HF tokenizer ID or local path (e.g. openai/gpt-oss-120b)
#   bridge_root     - Root of the Megatron-Bridge repo (default: script dir)
#   patch_script    - Path to export_custom_tokenizer_standalone.py (default: script dir)

set -euo pipefail

MEGATRON_PATH="${1:?Usage: $0 <megatron_path> <hf_path> <hf_model> <tokenizer_path> [bridge_root] [patch_script]}"
HF_PATH="${2:?missing hf_path}"
HF_MODEL="${3:?missing hf_model}"
TOKENIZER_PATH="${4:?missing tokenizer_path}"

BRIDGE_ROOT="${5:-../Megatron-Bridge}"
PATCH_SCRIPT="${6:-export_custom_tokenizer_standalone.py}"

CONVERT_SCRIPT="$BRIDGE_ROOT/examples/conversion/convert_checkpoints.py"

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "ERROR: convert_checkpoints.py not found at: $CONVERT_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$PATCH_SCRIPT" ]]; then
    echo "ERROR: patch script not found at: $PATCH_SCRIPT" >&2
    exit 1
fi

export PYTHONPATH="$BRIDGE_ROOT/3rdparty/Megatron-LM:$BRIDGE_ROOT/src:${PYTHONPATH:-}"

echo "========================================"
echo "Step 1/2: Export weights"
echo "  megatron : $MEGATRON_PATH"
echo "  hf model : $HF_MODEL"
echo "  output   : $HF_PATH"
echo "========================================"
python "$CONVERT_SCRIPT" export \
    --hf-model "$HF_MODEL" \
    --megatron-path "$MEGATRON_PATH" \
    --hf-path "$HF_PATH"

echo ""
echo "========================================"
echo "Step 2/2: Patch tokenizer and config"
echo "  tokenizer: $TOKENIZER_PATH"
echo "  output   : $HF_PATH"
echo "========================================"
python "$PATCH_SCRIPT" \
    --hf-path "$HF_PATH" \
    --tokenizer-path "$TOKENIZER_PATH"

echo ""
echo "Done."
echo "  Weights  : from $MEGATRON_PATH"
echo "  Config   : from $HF_MODEL (patched to match tokenizer)"
echo "  Tokenizer: from $TOKENIZER_PATH"
echo "  Output   : $HF_PATH"
