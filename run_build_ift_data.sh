#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
export HF_HOME="${CACHE_DIR}"

echo -e ">>> [IFT: Intentional Fine-Tuning - Training Data Preparation]"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "0.1" --least_num_per_domain "100" --valid_ratio "0.01"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "0.2" --least_num_per_domain "200" --valid_ratio "0.01"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "0.3" --least_num_per_domain "300" --valid_ratio "0.01"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "0.4" --least_num_per_domain "400" --valid_ratio "0.01"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "0.5" --least_num_per_domain "500" --valid_ratio "0.01"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01"

echo -e "\n\n\n"
echo -e ">>> [IFT: Intentional Fine-Tuning - Training Data Preparation] <<< DONE ALL"
