#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
from typing import Optional, List

import fire
import numpy as np

from utils.models import ModelUtils
from utils.data_io import DataIO
from utils.init_functions import logger_setup, cuda_setup, random_setup


class IFtDataBuilder:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "qwen3-8b",
            debug: bool = False,
            raw_data_dir: str = "data/intent_grasp/all/",
            save_dir: Optional[str] = None,
            **kwargs
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.model_name = model_name
        self.debug = debug
        self.kwargs = kwargs

        if not(isinstance(project_root_dir, str) and os.path.isdir(project_root_dir)):
            project_root_dir = os.getcwd()
        self.project_root_dir = project_root_dir
        assert os.path.isdir(self.project_root_dir)

        if not isinstance(raw_data_dir, str):
            raw_data_dir = os.path.join(self.project_root_dir, "data/intent_grasp/all/")
        assert os.path.isdir(raw_data_dir)
        self.raw_data_dir = raw_data_dir
        if not isinstance(save_dir, str):
            save_dir = os.path.join(self.project_root_dir, "data/ift_data")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Cache directory
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        # os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir

        self.tokenizer = None

        self.domain_id2name = {
            "DL": "daily life",
            "SA": "smart assistant",
            "TS": "toxic speech",
            "W": "writing",
            "G": "general",
            "EC": "e-commerce",
            "T": "teaching",
            "ER": "empathetic response",
            "N": "news",
            "CS": "customer support",
            "CP": "coronavirus pandemic",
            "PM": "policy making",
        }
        self.domain_name2id = {_v: _k for _k, _v in self.domain_id2name.items()}
        self.domain_id_all = list(self.domain_id2name.keys())
        self.domain_name_all = list(self.domain_id2name.values())

    def seq_len_stat(self, input_list: List[dict]) -> List[int]:
        if not isinstance(input_list, list) or len(input_list) == 0:
            return []

        if self.tokenizer is None:
            self.tokenizer = ModelUtils.initialize_tokenizer_hf(
                model_name=self.model_name, cache_dir=self.cache_dir,
                padding_side="left", truncation_side="left",
                verbose=self.verbose, model_ckpt_dir=None
            )

        # Tokenization
        input_tokens = [self.tokenizer.apply_chat_template(
            _item["messages"], return_tensors="pt", tokenize=True, return_dict=True) for _item in input_list]
        input_seq_len = [int(_tok["input_ids"].size(-1)) for _tok in input_tokens]

        # Do statistics
        self.logger.info(f">>> Data Statistics: total # token = {np.sum(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len max = {np.max(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len min = {np.min(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len avg = {np.mean(input_seq_len):.1f}")
        self.logger.info(f">>> Data Statistics: seq len std = {np.std(input_seq_len):.1f}")
        quantile_q = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
        quantiles = np.quantile(input_seq_len, q=quantile_q)
        quantiles = [float(np.round(quantile, 1)) for quantile in quantiles]
        self.logger.info(f">>> Data Statistics: seq len quantiles ({quantile_q}) = {quantiles}")

        del input_tokens
        return input_seq_len

    def seq_len_filter(
            self,
            input_list: List[dict],
            max_seq_len: int = 4096,
    ) -> List[dict]:
        if not isinstance(input_list, list) or len(input_list) == 0:
            return []

        if self.tokenizer is None:
            self.tokenizer = ModelUtils.initialize_tokenizer_hf(
                model_name=self.model_name, cache_dir=self.cache_dir,
                padding_side="left", truncation_side="left",
                verbose=self.verbose, model_ckpt_dir=None
            )

        # Tokenization
        input_tokens = [self.tokenizer.apply_chat_template(
            _item["messages"], return_tensors="pt", tokenize=True, return_dict=True) for _item in input_list]
        input_seq_len = [int(_tok["input_ids"].size(-1)) for _tok in input_tokens]
        assert len(input_seq_len) == len(input_list)

        # Do statistics
        self.logger.info(f">>> Data Statistics: total # token = {np.sum(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len max = {np.max(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len min = {np.min(input_seq_len):d}")
        self.logger.info(f">>> Data Statistics: seq len avg = {np.mean(input_seq_len):.1f}")
        self.logger.info(f">>> Data Statistics: seq len std = {np.std(input_seq_len):.1f}")
        quantile_q = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
        quantiles = np.quantile(input_seq_len, q=quantile_q)
        quantiles = [float(np.round(quantile, 1)) for quantile in quantiles]
        self.logger.info(f">>> Data Statistics: seq len quantiles ({quantile_q}) = {quantiles}")

        res_list = []
        for seq_len, input_item in zip(input_seq_len, input_list):
            if seq_len <= max_seq_len:
                res_list.append(input_item)
        self.logger.info(f">>> Filter [max_seq_len = {max_seq_len}]: "
                         f"len(input_list) = {len(input_list)} --> len(res_list) = {len(res_list)}")

        del input_tokens
        return res_list

    def build_ft_data(
            self,
            downsample_ratio: float = 1.0,
            least_num_per_domain: int = 100,
            valid_ratio: float = 0.01,
            max_seq_len: int = 4096,
            training_domains: str = "ALL",
            do_save: bool = True,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        assert 0 < downsample_ratio <= 1.0, f">>> !!! >>> Please set --downsample_ratio in the range of (0.0, 1.0]"
        assert least_num_per_domain > 0, f">>> !!! >>> Please set a positive value for --least_num_per_domain"
        assert 0 < valid_ratio <= 0.3, f">>> !!! >>> Please set --downsample_ratio in the range of (0.0, 0.3]"
        assert max_seq_len > 0, f">>> !!! >>> Please set a positive value for --max_seq_len"

        # Load the raw training data
        raw_train_fp = os.path.join(self.raw_data_dir, "train.jsonl")
        # raw_train_fp = os.path.join(self.raw_data_dir, "train.parquet")
        assert os.path.isfile(raw_train_fp)
        raw_train_data = DataIO.load_jsonl(raw_train_fp, verbose=True)
        # raw_train_data = DataIO.load_parquet(raw_train_fp, verbose=True)

        assert isinstance(raw_train_data, list) and len(raw_train_data) > 0
        num_train = len(raw_train_data)
        self.logger.info(f">>> num_train = {num_train}")

        # Deal with each instance
        ord_A = ord("A")
        show_cnt = int(1e4)
        ift_data_dict_full = dict()  # organized by domains
        for item_idx, item in enumerate(raw_train_data):
            assert isinstance(item, dict)

            cur_metadata = dict(item["metadata"])
            cur_domain = str(cur_metadata["domain_topic"][0][0])

            # cur_speaker = str(item["speaker"])
            cur_context = str(item["context"])
            cur_question = str(item["question"])
            cur_options = list(item["options"])
            cur_answer_intent = list(item["answer_intent"])
            cur_answer_index = list(item["answer_index"])
            assert len(cur_options) > 0
            assert len(cur_answer_intent) == len(cur_answer_index) > 0

            options_str = "\n".join(
                [f"({chr(ord_A + _op_idx)}) {_op_str}" for _op_idx, _op_str in enumerate(cur_options)]).strip()

            cur_sys_prompt = r"""
## Task: Your task is to answer the multiple-choice question. There could be one or more correct options.
## Requirements: You should put your final answer into "$\boxed{}$", such as $\boxed{A}$. \
If there are multiple answers, use commas to separate them, such as $\boxed{B, C}$.
            """.strip()
            cur_user_prompt = f"""
{cur_sys_prompt}

## Context:
{cur_context.strip()}

## Question:
{cur_question.strip()}
{options_str.strip()}
            """.strip()
            cur_user_dialog = [
                # {"role": "system", "content": cur_sys_prompt},
                {"role": "user", "content": cur_user_prompt}
            ]

            if len(cur_answer_intent) == 1:
                correct_answer_intent = str(cur_answer_intent[0]).lower().strip()
                correct_answer_label = chr(ord_A + int(cur_answer_index[0]))
                answer_analysis = f"The correct intent is {correct_answer_intent}"
                answer_final = r"\boxed{" + correct_answer_label + "}"
            else:
                correct_answer_intents = [str(_i).lower().strip() for _i in cur_answer_intent]
                correct_answer_labels = [chr(ord_A + int(_i)) for _i in cur_answer_index]
                answer_analysis = f"The correct intents include: " + " ".join(
                    [f"({_i_idx + 1}) {_i_str}" for _i_idx, _i_str in enumerate(correct_answer_intents)]).strip()
                answer_analysis = answer_analysis.strip()
                answer_final = r"\boxed{" + ", ".join(correct_answer_labels).strip() + "}"

            dialog_ift = cur_user_dialog + [{"role": "assistant", "content": f"""
{answer_analysis}

Final Answer: {answer_final}
            """.strip() + "\n"}]

            cur_item_ift = {
                "raw_info": item,
                "messages": dialog_ift,
            }

            if cur_domain not in ift_data_dict_full:
                ift_data_dict_full[cur_domain] = [cur_item_ift]
            else:
                ift_data_dict_full[cur_domain].append(cur_item_ift)

            if (item_idx + 1) % show_cnt == 0:
                self.logger.info(f">>> Progress: [Item: {item_idx + 1} / {num_train}]")

        # Done constructing IFT data. Optionally, conduct downsampling to each domain
        if 0 < downsample_ratio < 1.0:
            ift_data_dict = dict()
            for domain, data_list in ift_data_dict_full.items():
                assert isinstance(data_list, list) and len(data_list) > 0
                random.shuffle(data_list)  # To take random samples
                cur_num_data = len(data_list)
                cur_num_downsample = int(cur_num_data * downsample_ratio)
                if cur_num_downsample <= least_num_per_domain:
                    # Try to keep at least certain number of items per domain
                    cur_train_list = data_list[:least_num_per_domain]
                else:
                    # Take random samples of the current data list
                    cur_train_list = data_list[:cur_num_downsample]
                assert domain not in ift_data_dict
                ift_data_dict[domain] = cur_train_list
        else:
            ift_data_dict = ift_data_dict_full

        # Only keep certain domains for training
        if training_domains == "ALL":
            training_domains_str = "ALL"
        else:
            if isinstance(training_domains, str):
                if "," in training_domains:
                    training_domains = training_domains.split(",")
                else:
                    training_domains = [training_domains]
            elif isinstance(training_domains, list) or isinstance(training_domains, tuple):
                training_domains = list(training_domains)
            else:
                raise ValueError("--training_domains must be a string or a list")

            assert isinstance(training_domains, list) and len(training_domains) > 0
            training_domains_str = "_".join(training_domains)

            training_domains = [str(_td).strip() for _td in training_domains]
            assert all([_td in self.domain_id2name for _td in training_domains])
            training_domains = [self.domain_id2name[_td] for _td in training_domains]
            assert all([_td in ift_data_dict for _td in training_domains])
            training_domains_set = set(training_domains)

            ift_data_dict_select = dict()
            for domain, data_list in ift_data_dict.items():
                if domain in training_domains_set:
                    ift_data_dict_select[domain] = data_list
            ift_data_dict = ift_data_dict_select

        # Obtain the data list and split the train/valid sets
        ift_data_train_list = []
        ift_data_valid_list = []
        for domain, data_list in ift_data_dict.items():
            assert isinstance(data_list, list) and len(data_list) > 0
            cur_num_data = len(data_list)
            cur_num_valid = int(cur_num_data * valid_ratio)
            ift_data_train_list += data_list[cur_num_valid:]
            ift_data_valid_list += data_list[:cur_num_valid]
        assert len(ift_data_train_list) > 0 and len(ift_data_valid_list) > 0

        # Filter out samples that are longer than max_seq_len
        ift_data_train_list = self.seq_len_filter(ift_data_train_list, max_seq_len=max_seq_len)
        ift_data_valid_list = self.seq_len_filter(ift_data_valid_list, max_seq_len=max_seq_len)

        # Now, save the data
        if do_save:
            assert os.path.isdir(self.save_dir)
            if training_domains_str == "ALL":
                training_data_setting = (f"downsample_{downsample_ratio}--"
                                         f"least_{least_num_per_domain}--"
                                         f"valid_{valid_ratio}--"
                                         f"seq_{max_seq_len}")
            else:
                training_data_setting = (f"downsample_{downsample_ratio}--"
                                         f"least_{least_num_per_domain}--"
                                         f"valid_{valid_ratio}--"
                                         f"seq_{max_seq_len}--"
                                         f"domain_{training_domains_str}")
            cur_save_dir = os.path.join(self.save_dir, training_data_setting)
            os.makedirs(cur_save_dir, exist_ok=True)
            DataIO.save_jsonl(os.path.join(cur_save_dir, "train.jsonl"), ift_data_train_list, mode="w", verbose=True)
            DataIO.save_jsonl(os.path.join(cur_save_dir, "valid.jsonl"), ift_data_valid_list, mode="w", verbose=True)
            self.logger.info(f">>> Done saving. cur_save_dir: {cur_save_dir}")

        return None


def main(
    builder_task: int = 1,
    raw_data_dir: str = "data/intent_grasp/all/",
    downsample_ratio: float = 1.0,
    least_num_per_domain: int = 100,
    valid_ratio: float = 0.01,
    max_seq_len: int = 4096,
    training_domains: str = "ALL",
    model_name: str = "qwen3-8b",
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    save_dir: Optional[str] = None,
    **kwargs
) -> None:
    """
    :param builder_task: The process for the builder to run.
    :param raw_data_dir: The directory to the raw training data.
    :param downsample_ratio: The downsampling ratio of the training data.
    :param least_num_per_domain: The least number of training data per domain when considering downsampling.
    :param valid_ratio: The ratio of the validation set. I.e., valid / (valid + train)
    :param max_seq_len: The maximum sequence length of the training data (using the tokenizer of `model_name`).
    :param training_domains: The domains for training. ALL or DL,SA,TS,W,G,EC,T,ER,N,CS,CP,PM
    :param model_name: model name, e.g., "qwen3-8b"
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param save_dir: The dir path to save the results.

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Build_IFT_Data")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
    else:
        cache_dir = None

    ift_data_builder = IFtDataBuilder(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        model_name=model_name,
        debug=debug,
        raw_data_dir=raw_data_dir,
        save_dir=save_dir,
        **kwargs
    )

    builder_task = int(builder_task)
    logger.info(f">>> <START> [builder_task = {builder_task}]\n")
    match builder_task:
        case 1:
            ift_data_builder.build_ft_data(
                downsample_ratio=float(downsample_ratio),
                least_num_per_domain=int(least_num_per_domain),
                valid_ratio=float(valid_ratio),
                max_seq_len=int(max_seq_len),
                training_domains=training_domains,
                do_save=True,
            )
        case _:
            raise ValueError(f"ValueError: builder_task = {builder_task}")
    logger.info(f">>> <END> [builder_task = {builder_task}]\n\n\n")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
