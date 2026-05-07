#!/bin/bash

echo -e ">>> [IntentGrasp Evaluation Experiments - Proprietary LLMs: GPT-5, Gemini-3, Claude-4]"
for HF_ID in "gpt-5.4" "gpt-5.4-mini" "gpt-5.4-nano" "gemini-3.1-pro-preview" "gemini-3-flash-preview" "gemini-3.1-flash-lite-preview" "claude-opus-4-7" "claude-sonnet-4-6" "claude-haiku-4-5"
do
  # Evaluation on the down-sampled All Set
  for EVAL_SET in "all2gem_42" "all2gem_43" "all2gem_44"
  do
    SHUFFLE_SEED="-1"
    # Model Generation
    python3 run_gen_api.py --verbose \
      --cache_dir "${CACHE_DIR}" \
      --task_name "${EVAL_SET}" \
      --genai_model "${HF_ID}" \
      --seed_data "${SHUFFLE_SEED}" \
      --output_dir "results/gen_api_da-shuffle_${SHUFFLE_SEED}/"
    # Results Evaluation (F1 score)
    python3 run_eval_results.py --verbose \
      --cache_dir "${CACHE_DIR}" \
      --task_name "${EVAL_SET}" \
      --model_name "${HF_ID}" \
      --output_dir "results/gen_api_da-shuffle_${SHUFFLE_SEED}/"
  done

  # Evaluation on Gll Set
  for SHUFFLE_SEED in "-1" "7" "42" "365"
  do
    EVAL_SET="gem"
    # Model Generation
    python3 run_gen_api.py --verbose \
      --cache_dir "${CACHE_DIR}" \
      --task_name "${EVAL_SET}" \
      --genai_model "${HF_ID}" \
      --seed_data "${SHUFFLE_SEED}" \
      --output_dir "results/gen_api_da-shuffle_${SHUFFLE_SEED}/"
    # Results Evaluation (F1 score)
    python3 run_eval_results.py --verbose \
      --cache_dir "${CACHE_DIR}" \
      --task_name "${EVAL_SET}" \
      --model_name "${HF_ID}" \
      --output_dir "results/gen_api_da-shuffle_${SHUFFLE_SEED}/"
  done
done

echo -e "\n\n\n"
echo -e ">>> [IntentGrasp Evaluation Experiments - Proprietary LLMs] <<< DONE ALL"
