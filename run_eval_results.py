#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import string
from typing import Optional, List

import fire
import numpy as np

from utils.init_functions import logger_setup, cuda_setup, random_setup
from utils.data_io import DataIO


class LMEval:

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
            output_dir: Optional[str] = None,
            overwrite: bool = False,
            use_analysis: bool = False,
            # **kwargs
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.model_name = model_name
        self.debug = debug
        self.use_analysis = use_analysis
        # self.kwargs = kwargs

        if not(isinstance(project_root_dir, str) and os.path.isdir(project_root_dir)):
            project_root_dir = os.getcwd()
        self.project_root_dir = project_root_dir
        assert os.path.isdir(self.project_root_dir)

        self.output_dir = output_dir
        self.overwrite = overwrite

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
        self.terminators_gen = None
        self.model = None

        self.punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"

        self.special_re_token = r"<|-IFT-|>"

    @staticmethod
    def extract_boxed_answers(input_str: str) -> List[str]:
        boxed_answers_1 = re.findall(r"boxed{(.*?)}", input_str)
        boxed_answers_2 = re.findall(r"\$*\\boxed(.*?)\$", input_str)
        boxed_answers = []
        for b_ans in boxed_answers_1 + boxed_answers_2:
            if not isinstance(b_ans, str):
                continue
            b_ans = b_ans.strip()
            if len(b_ans) == 0:
                continue

            # boxed_answers.append(b_ans)
            # Consider the equivalent variants
            if len(b_ans) > 2 and b_ans.startswith("{") and b_ans.endswith("}"):
                boxed_answers.append(b_ans[1:-1])
            elif len(b_ans) > 2 and b_ans.startswith("(") and b_ans.endswith(")"):
                boxed_answers.append(b_ans[1:-1])
            elif len(b_ans) > 2 and b_ans.startswith("[") and b_ans.endswith("]"):
                boxed_answers.append(b_ans[1:-1])
            else:
                boxed_answers.append(b_ans)
                boxed_answers.append(b_ans.replace("\n", "").strip())

        boxed_answers = list(set(boxed_answers))

        return boxed_answers

    @staticmethod
    def normalize_text(raw_text: str) -> str:
        def _white_space(text: str):
            return " ".join(text.split())

        def _remove_articles(text: str):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def _remove_punc(text: str):
            punc_set = set(string.punctuation)
            return "".join(ch for ch in text if ch not in punc_set)

        cleaned_text = _white_space(_remove_articles(_remove_punc(raw_text.lower())))
        return cleaned_text

    def compute_score(
            self,
            prediction: str,
            references_idx: List[int],
            # **kwargs
    ) -> dict:
        # References
        ord_A = ord("A")
        cur_refs = [chr(ord_A + _ref_idx) for _ref_idx in references_idx]
        cur_refs = list(set(cur_refs))
        assert len(cur_refs) > 0

        prediction = str(prediction).strip()
        pred_all = []

        # Prediction: Extract boxed answers (also consider the cases where \boxed{} contains "\n")
        boxed_answers = self.extract_boxed_answers(prediction)
        boxed_answers_special = self.extract_boxed_answers(prediction.replace("\n", self.special_re_token))
        boxed_answers_special = [_ans.replace(self.special_re_token, "\n").strip() for _ans in boxed_answers_special]
        has_boxed_ans = len(boxed_answers) > 0 or len(boxed_answers_special) > 0
        pred_all += boxed_answers + boxed_answers_special
        pred_all = list(set(pred_all))

        pred_all = [str(_item).strip() for _item in pred_all if len(str(_item).strip()) > 0]
        if len(pred_all) == 0:
            return {"metric": "f1", "score": float(0.0),
                    "score_dict": {"f1": float(0.0), "p": float(0.0), "r": float(0.0)},
                    "refs": cur_refs, "preds": [], "has_boxed_ans": has_boxed_ans}
        elif len(pred_all) == 1:
            cur_pred = pred_all[0]
        else:
            # Use the shortest extraction
            pred_all = sorted(pred_all, key=lambda x: (len(x), x), reverse=False)
            cur_pred = pred_all[0]

        # Consider multiple choices (which should be separated by commas)
        if "," in cur_pred:
            cur_preds = [str(_item).strip() for _item in cur_pred.split(",")]
        else:
            cur_preds = [cur_pred]
        cur_preds = [_item.replace("(", "").replace(")", "").strip() for _item in cur_preds]
        cur_preds = list(set(cur_preds))
        if len(cur_preds) == 0:
            return {"metric": "f1", "score": float(0.0),
                    "score_dict": {"f1": float(0.0), "p": float(0.0), "r": float(0.0)},
                    "refs": cur_refs, "preds": [], "has_boxed_ans": has_boxed_ans}

        # Compute the precision score (ratio of the correct predictions among `cur_preds`)
        p_num_all = len(cur_preds)
        p_num_correct = 0
        for cur_pred in cur_preds:
            for cur_ref in cur_refs:
                pred_norm = self.normalize_text(cur_pred).strip()
                ref_norm = self.normalize_text(cur_ref).strip()
                if pred_norm == ref_norm:  # or pred_norm.startswith(ref_norm)
                    p_num_correct += 1
                    break
        assert p_num_all > 0
        p_score = float(p_num_correct / p_num_all)
        assert 0.0 <= p_score <= 1.0

        # Compute the recall score (ratio of the references being hit)
        r_num_all = len(cur_refs)
        r_num_correct = 0
        for cur_ref in cur_refs:
            for cur_pred in cur_preds:
                pred_norm = self.normalize_text(cur_pred).strip()
                ref_norm = self.normalize_text(cur_ref).strip()
                if pred_norm == ref_norm:  # or pred_norm.startswith(ref_norm)
                    r_num_correct += 1
                    break
        assert r_num_all > 0
        r_score = float(r_num_correct / r_num_all)
        assert 0.0 <= r_score <= 1.0

        if p_score + r_score == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2 * p_score * r_score / (p_score + r_score)
        assert 0.0 <= f1_score <= 1.0

        return {"metric": "f1", "score": float(f1_score),
                "score_dict": {"f1": float(f1_score), "p": float(p_score), "r": float(r_score)},
                "refs": cur_refs, "preds": cur_preds, "has_boxed_ans": has_boxed_ans}

    def lm_evaluate(
            self,
            task_name: str,
            gen_results: Optional[list] = None,
            do_save: bool = True,
    ) -> Optional[dict]:
        # Evaluation Phase: load result JSON, extract the reasoning/analysis and final answers, and compute scores

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Set the saving filepath
        assert isinstance(self.output_dir, str) and os.path.isdir(self.output_dir), "Please specify --output_dir"
        output_dir = os.path.join(self.output_dir, task_name, self.model_name)

        output_fn = "results_gen"
        output_fp = os.path.join(output_dir, output_fn + ".jsonl")
        output_eval_fp = os.path.join(output_dir, output_fn + "--eval" + ".json")
        if do_save:
            if os.path.isfile(output_eval_fp):
                if self.overwrite:
                    self.logger.info(f"Results will be overwritten: {output_eval_fp}")
                else:
                    self.logger.info(
                        f">>> model_name = {self.model_name}; output_dir: {output_dir}\n"
                        f">>> !!! >>> [SKIP; No --overwrite] File already exists: {output_eval_fp}"
                    )
                    return None
            else:
                self.logger.info(f"Results will be saved at: {output_eval_fp}")

        if not (isinstance(gen_results, list) and len(gen_results) > 0):
            if not os.path.isfile(output_fp):
                self.logger.info(
                    f">>> model_name = {self.model_name}; output_dir: {output_dir}\n"
                    f">>> !!! >>> [SKIP; No --output_fp] output_fp does not exist: {output_fp}"
                )
                return None

            if output_fp.endswith(".json"):
                gen_results = DataIO.load_json(output_fp, mode="r", verbose=True)
            elif output_fp.endswith(".jsonl"):
                gen_results = DataIO.load_jsonl(output_fp, mode="r", verbose=True)
            else:
                raise ValueError(f">>> !!! >>> Only JSON and JSONL are supported. output_fp: {output_fp}")

        # Deal with each task (and sub-tasks)
        self.logger.info(f">>> Evaluation Task: {task_name}")
        assert isinstance(gen_results, list) and len(gen_results) > 0, type(gen_results)
        num_results = len(gen_results)
        all_score_dicts = []
        all_score_values = []
        eval_metric = ""
        miss_pred_cnt_total = 0
        boxed_ans_cnt_total = 0
        has_boxed_ans_total = 0
        end_with_eot_cnt_total = 0
        is_end_with_eot_total = 0
        show_cnt = int(1e3)
        for item_idx, cur_res_dict in enumerate(gen_results):
            assert isinstance(cur_res_dict, dict) and len(cur_res_dict) > 0, type(cur_res_dict)

            # Load the attributes of the data item
            # metadata = dict(cur_res_dict["metadata"])  # the metadata of the dataset
            raw_info = dict(cur_res_dict["raw_info"])  # the data instance of IntentGrasp
            prediction = str(cur_res_dict["pred_answer"]).strip()  # model prediction to evaluate

            # task_type = "mcqa"
            assert "answer_index" in raw_info
            references = list(raw_info["answer_index"])
            references = [int(_item) for _item in references]

            if len(prediction) == 0:
                miss_pred = True
                cur_score_dict = {"metric": "f1", "score": float(0.0),
                                  "score_dict": {"f1": float(0.0), "p": float(0.0), "r": float(0.0)}}
            else:
                prediction = re.sub(r"[^\x00-\x7F]+", "", prediction).strip()  # remove non-ASCII
                cur_score_dict = self.compute_score(prediction=prediction, references_idx=references)
                assert "preds" in cur_score_dict and isinstance(cur_score_dict["preds"], list)
                miss_pred = len(cur_score_dict["preds"]) == 0

            assert "score" in cur_score_dict and "metric" in cur_score_dict
            cur_score_value = float(cur_score_dict["score"])
            cur_metric = str(cur_score_dict["metric"]).strip()
            assert len(cur_metric) > 0
            if len(eval_metric) == 0:
                eval_metric = cur_metric
            else:
                assert eval_metric == cur_metric, \
                    f">>> Assertion Error: `eval_metric` = {eval_metric}, but `cur_metric` = {cur_metric}"

            cur_res_dict["miss_pred"] = miss_pred
            if miss_pred:
                miss_pred_cnt_total += 1
            cur_res_dict["eval_dict"] = cur_score_dict
            cur_res_dict["eval_score"] = cur_score_value

            if "has_boxed_ans" in cur_score_dict:
                boxed_ans_cnt_total += 1
                assert isinstance(cur_score_dict["has_boxed_ans"], bool)
                if cur_score_dict["has_boxed_ans"]:
                    has_boxed_ans_total += 1

            if "end_with_eot" in cur_res_dict:
                end_with_eot_cnt_total += 1
                assert isinstance(cur_res_dict["end_with_eot"], bool)
                if cur_res_dict["end_with_eot"]:
                    is_end_with_eot_total += 1

            all_score_dicts.append(cur_res_dict)
            all_score_values.append(cur_score_value)

            if self.verbose and (item_idx + 1) % show_cnt == 0:
                self.logger.info(f">>> Progress: [Task: {task_name}] "
                                 f"[{item_idx + 1} / {num_results}] "
                                 f"[miss_pred: {miss_pred_cnt_total}] "
                                 f"[boxed_ans: {has_boxed_ans_total} / {boxed_ans_cnt_total}] "
                                 f"[end_with_eot: {is_end_with_eot_total} / {end_with_eot_cnt_total}]")

        # Compute the overall score statistics of different metrics and show stats
        num_items = len(all_score_values)
        match eval_metric:
            case "acc":
                # Each value is either 1.0 (correct) or 0.0 (incorrect)
                score_avg = float(np.mean(all_score_values).item())
            case "f1":
                # Each value is between 0.0 (totally wrong) to 1.0 (perfectly correct)
                score_avg = float(np.mean(all_score_values).item())
            case _:
                raise ValueError(f"ValueError: eval_metric = {eval_metric}")
        assert 0.0 <= score_avg <= 1.0, score_avg

        all_score_stat = {
            "num_items": num_items,
            "metric": eval_metric,
            "score_avg": score_avg,
        }
        all_scores = {
            "all_score_dicts": all_score_dicts,
            "all_score_values": all_score_values,
            "all_score_stat": all_score_stat,
        }

        self.logger.info(
            f">>> DONE ALL. [Task: {task_name}] [# = {num_items}] Overall Avg Score = {score_avg:.5f} "
            f"[miss_pred: {miss_pred_cnt_total}] "
            f"[boxed_ans: {has_boxed_ans_total} / {boxed_ans_cnt_total}] "
            f"[end_with_eot: {is_end_with_eot_total} / {end_with_eot_cnt_total}]")

        # Save the generation outputs
        if do_save:
            os.makedirs(output_dir, exist_ok=True)
            if output_eval_fp.endswith(".json"):
                DataIO.save_json(output_eval_fp, all_scores, mode="w", indent=2, verbose=True)
            elif output_eval_fp.endswith(".jsonl") and isinstance(all_scores, list):
                DataIO.save_jsonl(output_eval_fp, all_scores, mode="w", verbose=True)
            else:
                raise ValueError(f">>> !!! >>> Only JSON and JSONL are supported. output_eval_fp: {output_eval_fp}")

            self.logger.info(f">>> model_name = {self.model_name}; output_eval_fp: {output_eval_fp}")

        return all_scores


def main(
    task_name="",
    model_name: str = "qwen3-8b",
    cache_dir: Optional[str] = None,
    project_root_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    **kwargs
) -> None:
    """
    :param task_name: The name(s) of the IntentGrasp evaluation task(s). (e.g., "all", "gem", or "all,gem")
    :param model_name: LLM name, e.g., "qwen3-8b"
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param overwrite: Overwrite existing output files.

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Eval_Results")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
    else:
        cache_dir = None

    use_analysis = "use_analysis" in kwargs
    logger.info(f">>> [use_analysis: {use_analysis}]")

    lm_eval = LMEval(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_root_dir=project_root_dir,
        model_name=model_name,
        debug=debug,
        output_dir=output_dir,
        overwrite=overwrite,
        use_analysis=use_analysis,
        # **kwargs
    )

    if isinstance(task_name, str):
        task_name = [task_name]
    else:
        if isinstance(task_name, list) or isinstance(task_name, tuple):
            task_name = list(task_name)
        else:
            raise ValueError(f"--task_name should be a tuple/list/str: {task_name}")

    if isinstance(task_name, tuple) or isinstance(task_name, list):
        for cur_task_name in task_name:
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> {cur_task_name}\n")
            lm_eval.lm_evaluate(task_name=cur_task_name, gen_results=None, do_save=True)
            logger.info(f">>> <END> {cur_task_name}\n\n\n")
    else:
        raise ValueError(f"--task_name should be a tuple/list: {task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
