#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
export HF_HOME="${CACHE_DIR}"

BSZ=$2
if [[ -z ${BSZ} ]]; then
  BSZ="1"  # Set the batch size larger for faster generation
fi

echo -e ">>> [IntentGrasp Evaluation Experiments - Smaller language models]"
for HF_ID in "llama3-3b" "llama3-8b" "qwen3-4b" "qwen3-8b" "olmo3-7b" "gemma4-e2b" "gemma4-e4b"
do
  for EVAL_SET in "all" "gem"
  do
    for SHUFFLE_SEED in "-1" "7" "42" "365"
    do
      # Model Generation
      python3 run_gen_hf.py --verbose \
        --cache_dir "${CACHE_DIR}" \
        --task_name "${EVAL_SET}" \
        --model_name "${HF_ID}" \
        --seed_data "${SHUFFLE_SEED}" \
        --bsz "${BSZ}" --gen_config "0,0,0,0" --gen_method "da" \
        --output_dir "results/gen_hf_da-shuffle_${SHUFFLE_SEED}/"
      # Results Evaluation (F1 score)
      python3 run_eval_results.py --verbose --overwrite \
        --cache_dir "${CACHE_DIR}" \
        --task_name "${EVAL_SET}" \
        --model_name "${HF_ID}" \
        --output_dir "results/gen_hf_da-shuffle_${SHUFFLE_SEED}/"
    done
  done
done

echo -e "\n\n\n"
echo -e ">>> [IntentGrasp Evaluation Experiments - Larger language models]"  # (4-bit quantization)
for HF_ID in "llama3-70b" "qwen3-32b" "olmo3-32b" "gemma4-31b"
do
  for EVAL_SET in "all" "gem"
  do
    for SHUFFLE_SEED in "-1" "7" "42" "365"
    do
      # Model Generation
      python3 run_gen_hf.py --verbose \
        --cache_dir "${CACHE_DIR}" \
        --task_name "${EVAL_SET}" \
        --model_name "${HF_ID}" \
        --seed_data "${SHUFFLE_SEED}" \
        --bsz "${BSZ}" --gen_config "0,0,1,0" --gen_method "da" \
        --output_dir "results/gen_hf_da-shuffle_${SHUFFLE_SEED}/"
      # Results Evaluation (F1 score)
      python3 run_eval_results.py --verbose --overwrite \
        --cache_dir "${CACHE_DIR}" \
        --task_name "${EVAL_SET}" \
        --model_name "${HF_ID}" \
        --output_dir "results/gen_hf_da-shuffle_${SHUFFLE_SEED}/"
    done
  done
done

echo -e "\n\n\n"
echo -e ">>> [IntentGrasp Evaluation Experiments - Qwen3-4B and Qwen3-8B with baseline methods]"  # CoT & IA prompting
for HF_ID in "qwen3-4b" "qwen3-8b"
do
  for EVAL_SET in "all" "gem"
  do
    for SHUFFLE_SEED in "-1" "7" "42" "365"
    do
      for GEN_METHOD in "cot" "ia"
      do
        # Model Generation
        python3 run_gen_hf.py --verbose \
          --cache_dir "${CACHE_DIR}" \
          --task_name "${EVAL_SET}" \
          --model_name "${HF_ID}" \
          --seed_data "${SHUFFLE_SEED}" \
          --bsz "${BSZ}" --gen_config "0,0,0,0" --gen_method "${GEN_METHOD}" \
          --output_dir "results/gen_hf_${GEN_METHOD}-shuffle_${SHUFFLE_SEED}/"
        # Results Evaluation (F1 score)
        python3 run_eval_results.py --verbose --overwrite \
          --cache_dir "${CACHE_DIR}" \
          --task_name "${EVAL_SET}" \
          --model_name "${HF_ID}" \
          --output_dir "results/gen_hf_${GEN_METHOD}-shuffle_${SHUFFLE_SEED}/"
      done
    done
  done
done

echo -e "\n\n\n"
echo -e ">>> [IntentGrasp Evaluation Experiments - Open-source LLMs] <<< DONE ALL"
