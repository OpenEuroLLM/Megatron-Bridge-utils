# Megatron-Bridge-utils
Utility scripts for converting models with Megatron-Bridge

`export_with_custom_tokenizer_standalone.sh` takes four required arguments:

```
<megatron_path>   directory containing the Megatron checkpoint
<hf_path>         output directory for the exported HF model
<hf_model>        HF model ID for architecture reference (e.g. Qwen/Qwen3-0.6B)
<tokenizer_path>  HF tokenizer ID or local path to use in the exported model
```

## Example with Qwen3-0.6B and gpt-oss-120b tokenizer:

```
bash export_with_custom_tokenizer_standalone.sh \
        /path/to/megatron_checkpoint \
        /path/to/hf_output \
        Qwen/Qwen3-0.6B \
        openai/gpt-oss-120b
```