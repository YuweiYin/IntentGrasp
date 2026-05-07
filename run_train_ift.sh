#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
export HF_HOME="${CACHE_DIR}"
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"  # https://docs.wandb.ai/models/track/environment-variables

echo -e ">>> [IFT: Intentional Fine-Tuning - Training]"
for MODEL_NAME in "qwen3-4b" "qwen3-8b"
do
  # IFT using 10% of IntentGrasp training set
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_0.1--least_100--valid_0.01--seq_4096" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "100,1,0"

  # IFT using 20% of IntentGrasp training set
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_0.2--least_200--valid_0.01--seq_4096" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "200,1,0"

  # IFT using 30% of IntentGrasp training set
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_0.3--least_300--valid_0.01--seq_4096" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "300,1,0"

  # IFT using 40% of IntentGrasp training set
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_0.4--least_400--valid_0.01--seq_4096" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "400,1,0"

  # IFT using 50% of IntentGrasp training set
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_0.5--least_500--valid_0.01--seq_4096" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "500,1,0"

  # IFT using 100% of IntentGrasp training set
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"
done

echo -e "\n\n\n"
echo -e ">>> [IFT: Intentional Fine-Tuning - Training] <<< DONE ALL"
