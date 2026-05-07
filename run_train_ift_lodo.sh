#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
export HF_HOME="${CACHE_DIR}"
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"  # https://docs.wandb.ai/models/track/environment-variables

echo -e ">>> [IFT: Intentional Fine-Tuning - Leave-one-domain-out (Lodo) Training]"
for MODEL_NAME in "qwen3-4b" "qwen3-8b"
do
  # All 12 domains in IntentGrasp (DL_SA_TS_W_G_EC_T_ER_N_CS_CP_PM):
  #   daily life (DL), smart assistant (SA), toxic speech (TS), writing (W),
  #   general (G), e-commerce (EC), teaching (T), empathetic response (ER),
  #   news (N), customer support (CS), coronavirus pandemic (CP), and policy making (PM).

  # Lodo-IFT using 100% of IntentGrasp training set without the DL domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_SA_TS_W_G_EC_T_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the SA domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_TS_W_G_EC_T_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the TS domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_W_G_EC_T_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the W domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_G_EC_T_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the G domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_EC_T_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the EC domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_T_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the T domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_EC_ER_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the ER domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_EC_T_N_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the N domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_EC_T_ER_CS_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the CS domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_EC_T_ER_N_CP_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the CP domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_EC_T_ER_N_CS_PM" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"

  # Lodo-IFT using 100% of IntentGrasp training set without the PM domain
  python3 run_train_ift.py --verbose \
    --cache_dir "${CACHE_DIR}" \
    --model_name "${MODEL_NAME}" \
    --training_data_setting "downsample_1.0--least_1000--valid_0.01--seq_4096--domain_DL_SA_TS_W_G_EC_T_ER_N_CS_CP" \
    --train_mode "1,1,0" \
    --lora_mode "16,16,0" \
    --valid_mode "1000,1,0"
done

echo -e "\n\n\n"
echo -e ">>> [IFT: Intentional Fine-Tuning - Leave-one-domain-out (Lodo) Training] <<< DONE ALL"
