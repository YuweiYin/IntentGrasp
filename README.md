# IntentGrasp: A Comprehensive Benchmark for Intent Understanding

## Development Environments

<details><summary>Environment Setup</summary>

- **Server**: Linux (Ubuntu 22.04.5 LTS)
- **GPU**: NVIDIA CUDA GPU
  - (A6000 with 50GB VRAM or V100 with 32GB VRAM)
- **Python**: Python 3.10

```bash
# Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
conda create -n iu python=3.10 -y
conda activate iu

# Install packages for model generation/inference
pip install -r requirements.txt -i https://pypi.org/simple/
pip install -e . -i https://pypi.org/simple/

# Install packages for model training (GPU env)
pip install -r requirements_gpu.txt -i https://pypi.org/simple/
```

</details>


## IntentGrasp Benchmark

IntentGrasp is a large-scale, comprehensive, and standardized benchmark that evaluates intent understanding 
abilities across diverse domains and varying instance types.

* Download data from this anonymous [Kaggle dataset](https://www.kaggle.com/datasets/anonymous4kaggle/intentgrasp) and put it under the [`data/`](data/) directory.
  * Our IntentGrasp data adopts the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.
* Then, run `python3 run_data_loader.py` to load the data and check the data types & format for each data item.

<details><summary>IntentGrasp Data Directories</summary>

* [`data/`](data/)
  * IntentGrasp: [`data/intent_grasp/`](data/intent_grasp/)
    * IntentGrasp - All Set: [`data/intent_grasp/all/`](data/intent_grasp/all/)
      * `metadata.json`, `train.parquet`, `train.jsonl`, `test.parquet`, and `test.jsonl`
    * IntentGrasp - Gem Set: [`data/intent_grasp/gem/`](data/intent_grasp/gem/)
      * `metadata.json`, `test.parquet`, and `test.jsonl`

</details>


## IntentGrasp Evaluation

The IntentGrasp evaluation experiments demonstrate substantial room for LLMs to improve.

**Open-source LLM Evaluation**

```bash
# export HF_HOME="${HOME}/.cache/huggingface/"
CACHE_DIR="${HOME}/.cache/huggingface/"  # https://huggingface.co/docs/datasets/cache

# Download Open-source LLMs from Hugging Face
bash run_download_model_hf.sh "${CACHE_DIR}"

# Run Generation & Evaluation
BSZ="1"  # Set the batch size larger for faster generation
bash run_gen_hf.sh "${CACHE_DIR}" "${BSZ}"
```

**Proprietary LLM Evaluation**

```bash
# Set GenAI API keys as environment variables
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"  # https://platform.openai.com/settings/organization/api-keys
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"  # https://aistudio.google.com/app/apikey
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"  # https://platform.claude.com/docs/en/api/admin/api_keys/retrieve

# Downsample All Set (evaluation on the full All Set is costly)
python3 run_downsample_all_set.py --seed "42" --num_sample_subsets "3"

# Run Generation & Evaluation
bash run_gen_api.sh
```


## IFT: Intentional Fine-Tuning

**IFT Experiments**

The IFT training & evaluation experiments demonstrate the effectiveness of IFT 
in enhancing the intent understanding ability of LLMs.

```bash
# Set Wandb to monitor the training progress & validation scores
export WANDB_API_KEY="YOUR_WANDB_API_KEY"  # https://docs.wandb.ai/models/track/environment-variables
# export HF_HOME="${HOME}/.cache/huggingface/"
CACHE_DIR="${HOME}/.cache/huggingface/"  # https://huggingface.co/docs/datasets/cache

# IFT training data preparation
bash run_build_ift_data.sh "${CACHE_DIR}"

# Model Fine-tuning
# bash run_train_ift.sh "${CACHE_DIR}"
bash run_train_ift_unsloth.sh "${CACHE_DIR}"

# Model Evaluation after Fine-tuning:
#   After model fine-tuning, find the best checkpoint with the highest validation set score (based on Wandb records),
#   and then run generation & evaluation as in the previous section.
#   Please set `--model_ckpt_dir "ckpt/path/to/best/model/"` for run_gen_hf.py
```

**Lodo-IFT Experiments**

The Leave-one-domain-out (Lodo) experiments demonstrate the cross-domain generalizability of IFT.

```bash
# Set Wandb to monitor the training progress & validation scores
export WANDB_API_KEY="YOUR_WANDB_API_KEY"  # https://docs.wandb.ai/models/track/environment-variables
# export HF_HOME="${HOME}/.cache/huggingface/"
CACHE_DIR="${HOME}/.cache/huggingface/"  # https://huggingface.co/docs/datasets/cache

# Lodo-IFT training data preparation
bash run_build_ift_data_lodo.sh "${CACHE_DIR}"

# Model Fine-tuning
# bash run_train_ift_lodo.sh "${CACHE_DIR}"
bash run_train_ift_lodo_unsloth.sh "${CACHE_DIR}"

# Model Evaluation after Fine-tuning:
#   After model fine-tuning, find the best checkpoint with the highest validation set score (based on Wandb records),
#   and then run generation & evaluation as in the previous section.
#   Please set `--model_ckpt_dir "ckpt/path/to/best/model/"` for run_gen_hf.py
```


## License

* Our code uses the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license. Please refer to [LICENSE](./LICENSE) for more details.
* Our IntentGrasp data adopts the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.

---
