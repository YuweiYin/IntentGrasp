<div style="text-align: center;">

# IntentGrasp: A Comprehensive Benchmark for Intent Understanding

[![License: MIT](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) &nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2605.06832-b31b1b.svg)](https://arxiv.org/abs/2605.06832)

</div>

- **Paper:** https://arxiv.org/abs/2605.06832
  - **Authors:** [Yuwei Yin](https://www.yuweiyin.com/), [Chuyuan Li](https://chuyuanli.github.io/), [Giuseppe Carenini](https://www.cs.ubc.ca/~carenini/)
  - **Institute:** [UBC NLP Group](https://nlp.cs.ubc.ca/), Department of Computer Science, University of British Columbia
  - **Keywords:** Intent Understanding, Dataset, Benchmark, LLM, Evaluation, Intentional Fine-Tuning
- **Dataset:** https://huggingface.co/datasets/yuweiyin/IntentGrasp
  - Our IntentGrasp data adopts the [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.

<details open><summary>Paper Abstract</summary>

```text
Accurately understanding the intent behind speech, conversation, and writing is 
crucial to the development of helpful Large Language Model (LLM) assistants. 
This paper introduces IntentGrasp, a comprehensive benchmark for evaluating the 
intent understanding capability of LLMs. Derived from 49 high-quality, open-
licensed corpora spanning 12 diverse domains, IntentGrasp is constructed through 
source datasets curation, intent label contextualization, and task format 
unification. IntentGrasp contains a large-scale training set of 262,759 instances 
and two evaluation sets: an All Set of 12,909 test cases and a more balanced and 
challenging Gem Set of 470 cases. Extensive evaluations on 20 LLMs across 7 families 
(including frontier models such as GPT-5.4, Gemini-3.1-Pro, and Claude-Opus-4.7) 
demonstrate unsatisfactory performance, with scores below 60% on All Set and 
below 25% on Gem set. Notably, 17 out of 20 tested models perform worse than a 
random-guess baseline (15.2%) on Gem Set, while the estimated human performance 
is ~81.1%, showing substantial room for improvement. To enhance such ability, 
this paper proposes Intentional Fine-Tuning (IFT), which fine-tunes the models 
on the training set in IntentGrasp, yielding significant gains of 30+ F1 points 
on All Set and 20+ points on Gem Set. Tellingly, the leave-one-domain-out (Lodo) 
experiments further demonstrate the strong cross-domain generalizability of IFT, 
verifying that it is a promising approach to substantially enhancing the intent 
understanding of LLMs. Overall, by benchmarking and boosting intent understanding 
ability, this study sheds light on a promising path towards more intentional, 
capable, and safe AI assistants for human benefits and social good.
```

</details>


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

* Download data from Hugging Face dataset: https://huggingface.co/datasets/yuweiyin/IntentGrasp
  * Our IntentGrasp data adopts the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.

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


## Citation

* arXiv: https://arxiv.org/abs/2605.06832
* GitHub: https://github.com/YuweiYin/IntentGrasp
* Dataset: https://huggingface.co/datasets/yuweiyin/IntentGrasp

```bibtex
@article{yin2026intentgrasp,
  title   = {IntentGrasp: A Comprehensive Benchmark for Intent Understanding},
  author  = {Yin, Yuwei and Li, Chuyuan and Carenini, Giuseppe},
  journal = {arXiv preprint arXiv:2605.06832},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.06832}
}
```


## License

* Our code uses the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license. Please refer to [LICENSE](./LICENSE) for more details.
* Our IntentGrasp data adopts the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license.

---
