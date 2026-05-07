#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import re
import time
import random
from typing import Optional

import fire
import torch

from utils.models import ModelUtils
from utils.data_io import DataIO
from utils.prompting import PromptingMethods
from utils.init_functions import logger_setup, cuda_setup, random_setup


class LMGen:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            model_name: str = "qwen3-8b",
            model_ckpt_dir: Optional[str] = None,
            overwrite: bool = False,
            do_bf16: bool = False,
            do_4bit: bool = False,
            bsz: int = 1,
            debug: bool = False,
            output_dir: Optional[str] = None,
            max_new_gen: int = 2048,
            gen_temperature: float = 0.0,
            top_p: Optional[float] = None,
            top_k: Optional[float] = None,
            gen_method: str = "da",
            seed_data: Optional[int] = None,
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.model_name = model_name
        self.model_ckpt_dir = model_ckpt_dir
        self.do_bf16 = do_bf16
        self.do_4bit = do_4bit
        self.debug = debug
        self.overwrite = overwrite

        if not(isinstance(project_root_dir, str) and os.path.isdir(project_root_dir)):
            project_root_dir = os.getcwd()
        self.project_root_dir = project_root_dir
        assert os.path.isdir(self.project_root_dir)

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

        # Tokenizer and LLM model
        self.tokenizer = ModelUtils.initialize_tokenizer_hf(
            model_name=model_name, cache_dir=cache_dir, padding_side="left", truncation_side="left",
            verbose=verbose, model_ckpt_dir=self.model_ckpt_dir)
        self.terminators_gen = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        ]
        self.terminators_gen_set = set(self.terminators_gen)
        self.terminators_gen = list(self.terminators_gen_set)
        self.model = None

        # LM Generation settings
        self.output_dir = output_dir
        self.bsz = bsz
        self.max_new_gen = max_new_gen
        self.gen_temperature = gen_temperature
        self.top_p = top_p
        self.top_k = top_k
        self.gen_method = gen_method
        self.seed_data = seed_data

        # Set the filepath to the generator model
        model_path_local = ModelUtils.get_local_model_path(self.model_name, self.cache_dir)
        if isinstance(model_path_local, str) and os.path.isdir(model_path_local):
            self.model_path = model_path_local
        else:
            self.model_path = ModelUtils.OPEN_MODEL_HF[self.model_name]

    def run_inference(
            self,
            data_root_dir: str,
            task_name: str,
            eval_split: str = "test",
            num_to_eval: int = -1,
            data_start_idx: int = -1,
            data_end_idx: int = -1,
            show_cnt: int = 100,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert isinstance(self.output_dir, str), "Please specify --output_dir"

        if not (isinstance(data_root_dir, str) and os.path.isdir(data_root_dir)):
            data_root_dir = os.path.join(self.project_root_dir, "data")
        assert os.path.isdir(data_root_dir)
        data_dir = os.path.join(data_root_dir, "intent_grasp", task_name)
        assert os.path.isdir(data_dir)

        meta_filepath = os.path.join(data_dir, f"metadata.json")
        assert os.path.isfile(meta_filepath)
        metadata = DataIO.load_json(meta_filepath, verbose=self.verbose)
        assert isinstance(metadata, dict)

        data_filepath = os.path.join(data_dir, f"{eval_split}.jsonl")
        assert os.path.isfile(data_filepath)
        all_eval_data = DataIO.load_jsonl(data_filepath, verbose=self.verbose)
        assert isinstance(all_eval_data, list)

        len_before = len(all_eval_data)
        if 0 <= data_start_idx < data_end_idx and data_start_idx < len_before:
            do_divide_conquer = True
            all_eval_data = all_eval_data[data_start_idx: data_end_idx]
            self.logger.info(f">>> [idx range: {data_start_idx} -- {data_end_idx}] "
                             f"len(all_eval_data): {len_before} --> {len(all_eval_data)}")
            output_fn = f"results_gen--{data_start_idx}__{data_end_idx}"
        else:
            do_divide_conquer = False
            output_fn = "results_gen"

        # Set the output dir and filepath
        output_dir = os.path.join(self.output_dir, task_name, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_fp = os.path.join(output_dir, output_fn + ".jsonl")
        if os.path.isfile(output_fp):
            if self.overwrite:
                all_results = []
                done_ids = set()
                self.logger.info(f"Results will be overwritten: {output_fp}")
            else:
                # Load the previous outputs to resume running
                all_results = DataIO.load_jsonl(output_fp, mode="r+", verbose=True)

                if isinstance(all_results, list) and len(all_results) > 0:
                    done_ids = set([_res["item_id_key"] for _res in all_results])
                    self.logger.info(f"Resume running (len done = {len(done_ids)}): {output_fp}")
                else:
                    all_results = []
                    done_ids = set()
                    self.logger.info(f"Results will be saved at: {output_fp}")
        else:
            all_results = []
            done_ids = set()
            self.logger.info(f"Results will be saved at: {output_fp}")

        resume_from = len(done_ids)
        if resume_from > 0:
            all_eval_data = all_eval_data[resume_from:]
        len_dataset = len(all_eval_data)

        # Split the input list into mini batches
        assert isinstance(self.bsz, int) and self.bsz >= 1
        batches_data_items = [all_eval_data[_i: _i + self.bsz] for _i in range(0, len(all_eval_data), self.bsz)]
        num_batches = len(batches_data_items)

        # Load the generator model
        if self.model is None:
            if self.do_bf16:
                model = ModelUtils.initialize_model_hf(
                    model_name=self.model_name, cache_dir=self.cache_dir,
                    do_train=False, do_bf16=True, do_4bit=self.do_4bit, verbose=self.verbose,
                    model_ckpt_dir=self.model_ckpt_dir)
            else:
                model = ModelUtils.initialize_model_hf(
                    model_name=self.model_name, cache_dir=self.cache_dir,
                    do_train=False, do_fp16=True, do_4bit=self.do_4bit, verbose=self.verbose,
                    model_ckpt_dir=self.model_ckpt_dir)
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model = model

        self.logger.info(f">>> To Run: [Task: {task_name}] "
                         f"[#Batches = {num_batches} (BSZ={self.bsz})] [#Items = {len_dataset}]")

        if isinstance(self.seed_data, int) and self.seed_data > 0:
            do_shuffle = True
            random.seed(self.seed_data)
        else:
            do_shuffle = False

        item_idx = resume_from
        ord_A = ord("A")
        for batch_idx, cur_batch_data_items in enumerate(batches_data_items):
            assert isinstance(cur_batch_data_items, list) and len(cur_batch_data_items) > 0
            batch_prompts = []
            for cur_data_item in cur_batch_data_items:
                assert isinstance(cur_data_item, dict)

                assert "context" in cur_data_item and "question" in cur_data_item
                assert "answer_intent" in cur_data_item and "answer_index" in cur_data_item
                assert "options" in cur_data_item
                cur_context = str(cur_data_item["context"])
                cur_question = str(cur_data_item["question"])
                cur_options = list(cur_data_item["options"])

                if do_shuffle:  # Shuffle the options and update the correct intent labels/indices
                    cur_answer_intent = list(cur_data_item["answer_intent"])
                    assert len(cur_answer_intent) > 0
                    for _ans_intent in cur_answer_intent:
                        assert isinstance(_ans_intent, str) and _ans_intent in cur_options
                    random.shuffle(cur_options)
                    cur_answer_index = []
                    cur_answer_intent_set = set(cur_answer_intent)
                    for _op_idx, _op in enumerate(cur_options):
                        if _op in cur_answer_intent_set:
                            cur_answer_index.append(_op_idx)
                    assert len(cur_answer_intent) == len(cur_answer_index) > 0

                    cur_data_item["answer_index"] = cur_answer_index
                    cur_data_item["options"] = cur_options

                options_str = "\n".join(
                    [f"({chr(ord_A + _op_idx)}) {_op_str}" for _op_idx, _op_str in enumerate(cur_options)])

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

                cur_dialog = [
                    {"role": "user", "content": cur_user_prompt}
                ]
                if self.model_name.endswith("-base"):  # Base LLMs (text completion)
                    cur_prompt = "\n\n".join([str(_dialog["content"]).strip() for _dialog in cur_dialog]).strip()
                    cur_prompt += "\n\n## Answer:"
                else:  # Instruction-following LLMs
                    cur_prompt = self.tokenizer.apply_chat_template(
                        cur_dialog, tokenize=False, padding=False, return_tensors=None,
                        add_generation_prompt=True, enable_thinking=False,
                    )
                cur_prompt = str(cur_prompt).strip()
                if self.gen_method in PromptingMethods:
                    cur_prompt += "\n" + PromptingMethods[self.gen_method]
                cur_prompt = re.sub(r"[^\x00-\x7F]+", "", cur_prompt).strip()  # remove non-ASCII
                cur_prompt = cur_prompt.strip() + "\n"
                batch_prompts.append(cur_prompt)

            gen_results = ModelUtils.open_model_gen(
                inputs=batch_prompts, model=self.model, tokenizer=self.tokenizer, need_tokenize=True,
                max_new_tokens=self.max_new_gen,
                temperature=self.gen_temperature, top_p=self.top_p, top_k=self.top_k,
            )
            assert len(gen_results) == len(batch_prompts), len(gen_results)
            batch_pred_answer = [str(gen_result["output_text"]).strip() for gen_result in gen_results]
            batch_eot = [bool(gen_result["end_with_eot"]) for gen_result in gen_results]

            break_flag = False
            for eval_item, prompt, pred_answer, cur_eot in zip(
                    cur_batch_data_items, batch_prompts, batch_pred_answer, batch_eot):
                item_id_key = f"{item_idx}"
                pred_answer = re.sub(r"[^\x00-\x7F]+", "", pred_answer).strip()  # remove non-ASCII
                if "original_task" in eval_item:
                    original_task = str(eval_item["original_task"])
                    if original_task in metadata:
                        metadata_to_save = metadata[original_task]
                    else:
                        metadata_to_save = metadata
                else:
                    assert "metadata" in eval_item
                    metadata_to_save = eval_item["metadata"]
                cur_gen_output = {
                    "metadata": metadata_to_save,
                    "raw_info": eval_item,
                    "method": self.gen_method,
                    "model": self.model_name,
                    "item_id_key": item_id_key,
                    "ds_id": "",  # ds_id
                    "batch_idx": batch_idx,
                    "item_idx": item_idx,
                    "prompt": prompt,  # The input prompt
                    "pred_answer": pred_answer,
                    "output_text": pred_answer,
                    "end_with_eot": bool(cur_eot),  # True if the output ends with end-of-text
                }
                item_idx += 1

                all_results.append(cur_gen_output)
                if item_idx % show_cnt == 0:
                    cur_log_info = (f">>> Progress: [Task: {task_name}] "
                                    f"[Batch (size={self.bsz}): {batch_idx + 1} / {num_batches}] "
                                    f"[Item: {item_idx} / {len_dataset + resume_from}]")
                    if do_divide_conquer:
                        cur_log_info += f" [idx range: {data_start_idx} -- {data_end_idx}]"
                    if num_to_eval > 0:
                        cur_log_info += f" [num_to_eval = {num_to_eval}]"
                    if resume_from > 0:
                        cur_log_info += f" [resume_from = {resume_from}]"
                    if self.verbose:
                        self.logger.info(cur_log_info)

                    DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=False)
                    gc.collect()
                    torch.cuda.empty_cache()

                if item_idx >= num_to_eval > 0:
                    break_flag = True
                    break

            if break_flag:
                break

        # Show logs and save the results
        if self.verbose:
            self.logger.info(
                f">>> Done. [Task: {task_name}] # = {len_dataset} "
                f"[num_to_eval = {num_to_eval}] [resume_from = {resume_from}] "
                f"[idx range: {data_start_idx} -- {data_end_idx}]")
        DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=True)
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(
            f">>> DONE ALL. model_name = {self.model_name}\n"
            f"gen_temperature: {self.gen_temperature}, batch_size: {self.bsz}"
        )


def main(
    data_root_dir: Optional[str] = None,
    task_name: Optional[str] = None,
    eval_split: str = "test",
    model_name: str = "qwen3-8b",
    model_ckpt_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    bsz: int = 1,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    max_new_gen: int = 2048,
    gen_temperature: float = 0.0,
    gen_top_p: Optional[float] = -1.0,
    gen_config: Optional[str] = None,
    gen_method: str = "da",
    num_to_eval: int = -1,
    data_start_idx: int = -1,
    data_end_idx: int = -1,
    seed_data: Optional[int] = None,
    show_cnt: int = 100,
    **kwargs
) -> None:
    """
    :param data_root_dir: The directory to load/store the data.
    :param task_name: The name(s) of the IntentGrasp evaluation task(s). (e.g., "all", "gem", or "all,gem")
    :param eval_split: The data split to evaluation (train/valid/test).
    :param model_name: LLM name, e.g., "qwen3-8b"
    :param model_ckpt_dir: Load checkpoint from this directory. (It overwrites the effect of `model_name`)
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param bsz: The batch size.
    :param verbose: Verbose mode: show logs.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param max_new_gen: The maximum number of newly generated tokens.
    :param gen_temperature: The temperature used in LLM generation. Default: 0.
    :param gen_top_p: The Top-p ratio used in LLM generation.
    :param gen_config: The LLM generation configurations.
    :param gen_method: The method/baseline to use.
    :param num_to_eval: The total number of instances to evaluate. -1 means using all.
    :param data_start_idx: The start index of the test data (divide and conquer) per subtask.
    :param data_end_idx: The ending index of the test data (divide and conquer) per subtask.
    :param seed_data: The random seed to shuffle the options for multiple-choice QA. None: no shuffling
    :param show_cnt: Show the logs and save the results per `show_cnt` number of instances.

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Gen_HF")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
    else:
        cache_dir = None

    # Parse the `gen_config` argument
    if (isinstance(gen_config, tuple) or isinstance(gen_config, list) or
            (isinstance(gen_config, str) and len(gen_config.strip()) > 0)):
        if isinstance(gen_config, tuple):
            gen_config = list(gen_config)
        if isinstance(gen_config, str):
            gen_config = gen_config.strip()
            gen_config = gen_config.split(",")
        # Note: For boolean parameters, "1" means True and "0" means False
        # overwrite: Whether to overwrite existing output files.
        # do_bf16: Whether to use BF16 precision mode to load models.
        # do_4bit: Whether to use 4bit quantization mode to load models.
        # debug: Debugging / developing mode.
        overwrite = str(gen_config[0]) == "1"
        do_bf16 = str(gen_config[1]) == "1"
        do_4bit = str(gen_config[2]) == "1"
        debug = str(gen_config[3]) == "1"
    else:
        gen_config = ["0" for _ in range(4)]  # default: All "0"s
        overwrite = do_bf16 = do_4bit = debug = False
    logger.info(f">>> gen_config = {gen_config}: [overwrite: {overwrite}] "
                f"[do_bf16: {do_bf16}] [do_4bit: {do_4bit}] [debug: {debug}]")

    lm_gen = LMGen(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        model_name=model_name,
        model_ckpt_dir=model_ckpt_dir if (isinstance(model_ckpt_dir, str) and len(model_ckpt_dir) > 0) else None,
        bsz=max(int(bsz), 1),
        overwrite=overwrite,
        do_bf16=do_bf16,
        do_4bit=do_4bit,
        debug=debug,
        output_dir=output_dir,
        max_new_gen=max(int(max_new_gen), 128),
        gen_temperature=float(gen_temperature),
        top_p=float(gen_top_p) if float(gen_top_p) > 0.0 else None,
        top_k=None,
        gen_method=gen_method,
        seed_data=seed_data,
    )

    if isinstance(task_name, str):
        task_name = [task_name]
    elif isinstance(task_name, list) or isinstance(task_name, tuple):
        task_name = list(task_name)
    else:
        task_name = ["all", "gem"]

    if isinstance(task_name, tuple) or isinstance(task_name, list):
        for cur_task_name in task_name:
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> {cur_task_name}\n")
            lm_gen.run_inference(
                data_root_dir=data_root_dir, task_name=cur_task_name, eval_split=eval_split,
                num_to_eval=int(num_to_eval), data_start_idx=int(data_start_idx), data_end_idx=int(data_end_idx),
                show_cnt=max(10, int(show_cnt)),
            )
            logger.info(f">>> <END> {cur_task_name}\n\n\n")
    else:
        raise ValueError(f"--task_name should be a tuple/list/str: {task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
