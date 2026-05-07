#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
export HF_HOME="${CACHE_DIR}"

echo -e ">>> [Lodo-IFT: Leave-one-domain-out IFT - Training Data Preparation]"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "SA,TS,W,G,EC,T,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,TS,W,G,EC,T,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,W,G,EC,T,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,G,EC,T,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,EC,T,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,T,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,EC,ER,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,EC,T,N,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,EC,T,ER,CS,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,EC,T,ER,N,CP,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,EC,T,ER,N,CS,PM"

python3 run_build_ift_data.py --verbose --cache_dir "${CACHE_DIR}" \
  --model_name "qwen3-8b" --raw_data_dir "data/intent_grasp/all/" --save_dir "data/ift_data" \
  --downsample_ratio "1.0" --least_num_per_domain "1000" --valid_ratio "0.01" \
  --training_domains "DL,SA,TS,W,G,EC,T,ER,N,CS,CP"

echo -e "\n\n\n"
echo -e ">>> [Lodo-IFT: Leave-one-domain-out IFT - Training Data Preparation] <<< DONE ALL"
