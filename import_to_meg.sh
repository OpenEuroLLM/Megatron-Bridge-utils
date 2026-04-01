HF_MODEL="${1:?Usage: $0 <hf_model> <megatron_export_path> [bridge_root]}"
MEGATRON_PATH="${2:?missing megatron export path}"
BRIDGE_ROOT="${3:-../Megatron-Bridge}"

CONVERT_SCRIPT="$BRIDGE_ROOT/examples/conversion/convert_checkpoints.py"

export PYTHONPATH="$BRIDGE_ROOT/3rdparty/Megatron-LM:$BRIDGE_ROOT/src:${PYTHONPATH:-}"

python $BRIDGE_ROOT/examples/conversion/convert_checkpoints.py import --hf-model $HF_MODEL --megatron-path $MEGATRON_PATH
