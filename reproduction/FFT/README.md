# FFT

This folder contains assets for **Full Fine-Tuning (FFT)** experiments.

## Folder structure

- `extension/`: supporting scripts or configs for FFT extensions.
- `finetuning/`: main FFT training artifacts and references.
- `limited-data-exp/`: limited-data FFT experiments.
- `malicious-exp/`: malicious-data FFT experiments.

## FFT-specific hyperparameters

- Learning rate: $5 \times 10^{-6}$
- Batch size: $8$

## Full Parameter Fine Tuning (FFT)

Example command for Qwen2-7B-Instruct:

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset beaver_train330k_privacy_safe_1k,beaver_train330k_fairness_safe_1k \
    --cutoff_len 2048 \
    --learning_rate 5e-06 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0.1 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir saves/Qwen2-7B-Instruct/full/train_2026-01-17-01-16-00 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_bnb_8bit
```

Note: Due to hardware limitations the optimizer was changed to `adamw_bnb_8bit` from `adamw_torch`.

## FFT smoke test

```bash
python -m llamafactory.cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path Qwen/Qwen2-7B-Instruct \
  --finetuning_type full \
  --template qwen \
  --trust_remote_code True \
  \
  --dataset_dir data \
  --dataset beaver_train330k_fairness_safe_1k \
  \
  --cutoff_len 128 \
  --max_samples 2 \
  \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  \
  --learning_rate 5e-06 \
  --lr_scheduler_type cosine \
  --optim adamw_bnb_8bit \
  \
  --logging_steps 1 \
  --save_steps 20 \
  \
  --packing False \
  --enable_thinking False \
  --flash_attn disabled \
  \
  --bf16 True \
  --report_to none \
  \
  --output_dir saves/smoke_test/qwen2_7b_full
```
