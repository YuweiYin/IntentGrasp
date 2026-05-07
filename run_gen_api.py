#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import re
import time
import random
from typing import Optional

import fire

from utils.models import ModelUtils
from utils.data_io import DataIO
from utils.prompting import PromptingMethods
from utils.init_functions import logger_setup, cuda_setup, random_setup


class GenAI:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: Optional[dict] = None,
            seed: int = 42,
            eval_task_name: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            genai_model: str = "gpt-5.4-nano",
            genai_api_key: Optional[str] = None,
            debug: bool = False,
            overwrite: bool = False,
            output_dir: Optional[str] = None,
            max_new_gen: int = 2048,
            gen_temperature: float = 1.0,
            gen_method: str = "da",
            seed_data: Optional[int] = None,
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.debug = debug
        self.overwrite = overwrite

        if not(isinstance(project_root_dir, str) and os.path.isdir(project_root_dir)):
            project_root_dir = os.getcwd()
        self.project_root_dir = project_root_dir
        assert os.path.isdir(self.project_root_dir)

        self.eval_task_name = eval_task_name
        self.output_dir = output_dir
        self.max_new_gen = max_new_gen
        self.gen_temperature = gen_temperature
        self.gen_method = gen_method
        self.seed_data = seed_data

        # GenAI settings
        self.genai_model = genai_model
        if isinstance(genai_api_key, str) and len(genai_api_key) > 0:
            self.genai_api_key = genai_api_key
        else:
            if "gpt" in genai_model:
                self.genai_api_key = os.getenv("OPENAI_API_KEY")
            elif "gemini" in genai_model:
                self.genai_api_key = os.getenv("GEMINI_API_KEY")
            elif "claude" in genai_model:
                self.genai_api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                raise ValueError(f">>> Unsupported genai model: {genai_model}")

    def run_inference(
            self,
            data_root_dir: str,
            task_name: str,
            eval_split: str = "test",
            num_to_eval: int = -1,
            data_start_idx: int = -1,
            data_end_idx: int = -1,
            show_cnt: int = 10,
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
        output_dir = os.path.join(self.output_dir, task_name, self.genai_model)
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

        if isinstance(self.seed_data, int) and self.seed_data > 0:
            do_shuffle = True
            random.seed(self.seed_data)
        else:
            do_shuffle = False

        item_idx = resume_from
        ord_A = ord("A")
        for cur_data_item in all_eval_data:
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

            system_prompt = r"""
## Task: Your task is to answer the multiple-choice question. There could be one or more correct options.
## Requirements: You should put your final answer into "$\boxed{}$", such as $\boxed{A}$. \
If there are multiple answers, use commas to separate them, such as $\boxed{B, C}$.
            """.strip()
            user_prompt = f"""
## Context:
{cur_context.strip()}

## Question:
{cur_question.strip()}
{options_str.strip()}
            """.strip()

            if self.gen_method in PromptingMethods:
                cur_prompting = PromptingMethods[self.gen_method].strip()
                if len(cur_prompting) > 0:
                    system_prompt += " " + cur_prompting.replace("Let's", "In your analysis, you must")
                    user_prompt += "\n\n" + cur_prompting.replace("Let's", "You must")
            user_prompt = user_prompt.strip() + "\n"

            # Send GenAI request
            if "gpt" in self.genai_model:
                gpt_input_messages = [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = ModelUtils.call_gpt(
                    openai_model_name=self.genai_model, messages=gpt_input_messages,
                    openai_api_key=self.genai_api_key,
                )  # format_class=format_class,
                res_message = response.choices[0].message
                refusal = res_message.refusal
                if refusal:  # If the model refuses to respond, get the refusal message
                    self.logger.info(f">>> !!! >>> The model refuses to respond: {refusal}")
                output_text = str(res_message.content).strip()
            elif "gemini" in self.genai_model:
                gemini_input_messages = [system_prompt, user_prompt]
                response = ModelUtils.call_gemini(
                    gemini_model_name=self.genai_model, messages=gemini_input_messages,
                    gemini_api_key=self.genai_api_key,
                )
                res_message = response.text
                output_text = str(res_message).strip()
            elif "claude" in self.genai_model:
                claude_input_messages = [system_prompt, user_prompt]
                response = ModelUtils.call_claude(
                    claude_model_name=self.genai_model, messages=claude_input_messages,
                    claude_api_key=self.genai_api_key,
                    max_output_tokens=self.max_new_gen,
                )
                try:
                    res_message = response.content[0].text
                except Exception as e:
                    self.logger.info(e)
                    res_message = "NONE"
                output_text = str(res_message).strip()
            else:
                raise ValueError(f">>> Unsupported genai model: {self.genai_model}")

            item_id_key = f"{item_idx}"
            pred_answer = re.sub(r"[^\x00-\x7F]+", "", output_text).strip()  # remove non-ASCII
            if "original_task" in cur_data_item:
                original_task = str(cur_data_item["original_task"])
                if original_task in metadata:
                    metadata_to_save = metadata[original_task]
                else:
                    metadata_to_save = metadata
            else:
                assert "metadata" in cur_data_item
                metadata_to_save = cur_data_item["metadata"]
            cur_gen_output = {
                "metadata": metadata_to_save,
                "raw_info": cur_data_item,
                "method": self.gen_method,
                "model": self.genai_model,
                "item_id_key": item_id_key,
                "ds_id": "",  # ds_id
                "batch_idx": "",
                "item_idx": item_idx,
                "prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },  # The input prompt
                "pred_answer": pred_answer,
                "output_text": pred_answer,
            }
            item_idx += 1

            all_results.append(cur_gen_output)
            # done_ids.add(item_id_key)
            if item_idx % show_cnt == 0:
                cur_log_info = (f">>> Progress: [Task: {task_name}] "
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

            if item_idx >= num_to_eval > 0:
                break

        # Show logs and save the results
        if self.verbose:
            self.logger.info(
                f">>> Done. [Task: {task_name}] # = {len_dataset} "
                f"[num_to_eval = {num_to_eval}] [resume_from = {resume_from}] "
                f"[idx range: {data_start_idx} -- {data_end_idx}]")
        DataIO.save_jsonl(output_fp, all_results, mode="w", verbose=True)
        gc.collect()
        self.logger.info(
            f">>> DONE ALL. genai_model = {self.genai_model}\n"
            f"gen_temperature: {self.gen_temperature}, batch_size=1"
        )


def main(
    data_root_dir: Optional[str] = None,
    task_name: Optional[str] = None,
    eval_split: str = "test",
    project_root_dir: Optional[str] = None,
    genai_model: str = "gpt-5.4-nano",
    genai_api_key: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    overwrite: bool = False,
    output_dir: Optional[str] = None,
    max_new_gen: int = 2048,
    gen_temperature: float = 1.0,
    gen_method: str = "da",
    num_to_eval: int = -1,
    data_start_idx: int = -1,
    data_end_idx: int = -1,
    seed_data: Optional[int] = None,
    show_cnt: int = 10,
    **kwargs
) -> None:
    """
    :param data_root_dir: The directory to load/store the data.
    :param task_name: The name(s) of the IntentGrasp evaluation task(s). (e.g., "all", "gem", or "all,gem")
    :param eval_split: The data split to evaluation (train/valid/test).
    :param project_root_dir: The root directory of the current project/repo.
    :param genai_model: e.g., "gpt-5.4", "claude-opus-4.6", or "gemini-3.1-pro-preview"
    :param genai_api_key: your valid API Key (OpenAI or Gemini). Default: env var ${OPENAI_API_KEY} or ${GEMINI_API_KEY}
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param overwrite: Whether to overwrite existing output files.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param max_new_gen: The maximum number of newly generated tokens.
    :param gen_temperature: The temperature used in LLM generation. Default: 1.0
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
    logger = logger_setup("LM_Gen_API")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    api_gen = GenAI(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        project_root_dir=project_root_dir,
        genai_model=genai_model,
        genai_api_key=genai_api_key,
        debug=debug,
        overwrite=overwrite,
        output_dir=output_dir,
        max_new_gen=max(int(max_new_gen), 128),
        gen_temperature=float(gen_temperature),
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
            api_gen.run_inference(
                data_root_dir=data_root_dir,
                task_name=cur_task_name, eval_split=eval_split,
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
