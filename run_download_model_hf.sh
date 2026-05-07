#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
export HF_HOME="${CACHE_DIR}"

YOUR_HF_TOKEN=$2
if [[ -z ${YOUR_HF_TOKEN} ]]; then
  YOUR_HF_TOKEN="${HF_TOKEN}"  # https://huggingface.co/settings/tokens
fi

# Llama3
for HF_ID in "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct"
do
  python3 utils/download_hf_model.py \
    --trust_remote_code --verbose \
    --cache_dir "${CACHE_DIR}" \
    --hf_token "${YOUR_HF_TOKEN}" \
    --hf_id "${HF_ID}"
done

# Qwen3
for HF_ID in "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-32B"
do
  python3 utils/download_hf_model.py \
    --trust_remote_code --verbose \
    --cache_dir "${CACHE_DIR}" \
    --hf_token "${YOUR_HF_TOKEN}" \
    --hf_id "${HF_ID}"
done

# Olmo3
for HF_ID in "allenai/Olmo-3-7B-Instruct" "allenai/Olmo-3.1-32B-Instruct"
do
  python3 utils/download_hf_model.py \
    --trust_remote_code --verbose \
    --cache_dir "${CACHE_DIR}" \
    --hf_token "${YOUR_HF_TOKEN}" \
    --hf_id "${HF_ID}"
done

# Gemma4
for HF_ID in "google/gemma-4-E2B-it" "google/gemma-4-E4B-it" "google/gemma-4-31B-it"
do
  python3 utils/download_hf_model.py \
    --trust_remote_code --verbose \
    --cache_dir "${CACHE_DIR}" \
    --hf_token "${YOUR_HF_TOKEN}" \
    --hf_id "${HF_ID}"
done
