#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import logging
from typing import Optional

import fire

from utils.init_functions import logger_setup, random_setup
from utils.data_io import DataIO


class TaskManager:

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        self.verbose = verbose
        if logger is None:
            self.logger = logging.getLogger("TaskManager")
        else:
            self.logger = logger

        if isinstance(project_root_dir, str) and os.path.isdir(project_root_dir):
            self.project_root_dir = project_root_dir
        else:
            self.project_root_dir = os.getcwd()
        assert os.path.isdir(project_root_dir)

        if isinstance(data_dir, str) and os.path.isdir(data_dir):
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(self.project_root_dir, "data")
        assert os.path.isdir(self.data_dir)
        self.raw_data_dir = os.path.join(self.data_dir, "raw_data")
        self.unified_data_dir = os.path.join(self.data_dir, "unified_data")
        assert os.path.isdir(self.raw_data_dir), f"Raw data dir not found: {self.raw_data_dir}"
        os.makedirs(self.unified_data_dir, exist_ok=True)

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

        # os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = self.cache_dir

        self.task_name = None
        self.task_meta = None
        self.task_data = None

        self.intent_label2statement = dict()
        self.num_options = 10  # The max number of options

    def load_task(
            self,
            do_save: bool = False,
    ) -> None:
        self.raw_data_unification(do_save=do_save)
        if self.task_name != "dyndst":
            self.mcqa_construction(do_save=do_save)
        return None

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        raise NotImplementedError

    def mcqa_construction(
            self,
            do_save: bool = False,
    ) -> None:
        assert isinstance(self.task_meta, dict)
        for task_split in ["train", "valid", "test"]:
            cur_data = self.task_data[task_split]["data"]
            assert isinstance(cur_data, list)
            processed_items = []
            for cur_item in cur_data:
                assert isinstance(cur_item, dict)
                assert "answer_intent" in cur_item
                answer_intent = list(cur_item["answer_intent"])
                answer_intent_set = set(answer_intent)

                # Get distractors via random sampling (from other intent classes) [and/or intent statement ranking]
                all_intents_set = set(self.intent_label2statement.values())
                assert len(all_intents_set) > 0
                for _intent in answer_intent:
                    assert isinstance(_intent, str) and _intent in all_intents_set
                if len(all_intents_set) <= self.num_options:
                    all_intents_list = list(all_intents_set)
                    assert 0 < len(all_intents_list) <= self.num_options
                else:
                    wrong_intents_set = all_intents_set - set(answer_intent)
                    assert len(wrong_intents_set) + len(answer_intent) >= self.num_options
                    wrong_intents_list = random.sample(wrong_intents_set, self.num_options - len(answer_intent))
                    all_intents_list = answer_intent + wrong_intents_list
                    assert 0 < len(all_intents_list) == self.num_options

                random.shuffle(all_intents_list)
                answer_index = []
                for _idx, _intent in enumerate(all_intents_list):
                    if _intent in answer_intent_set:
                        answer_index.append(_idx)
                assert len(answer_index) == len(answer_intent) > 0
                cur_item["answer_index"] = answer_index
                cur_item["options"] = all_intents_list

                # Check data types & format
                assert isinstance(cur_item["paper_year"], int) and cur_item["paper_year"] > 0
                cur_domain_topic = cur_item["domain_topic"]
                assert isinstance(cur_domain_topic, list) and len(cur_domain_topic) > 0
                for d_t in cur_domain_topic:
                    assert isinstance(d_t, list) and len(d_t) == 2
                cur_answer_intent = cur_item["answer_intent"]  # there must be at least one correct intent answer
                assert isinstance(cur_answer_intent, list) and len(cur_answer_intent) > 0
                cur_answer_index = cur_item["answer_index"]  # the answer index can one-to-one map to answer intent
                assert isinstance(cur_answer_index, list) and len(cur_answer_index) == len(cur_answer_intent) > 0
                cur_options = cur_item["options"]  # the number of options must be larger than # of correct intents
                assert isinstance(cur_options, list) and len(cur_options) > len(cur_answer_intent) > 0
                cur_options_set = set(cur_options)
                for _intent in cur_answer_intent:  # the correct intent answer must be in the options list
                    assert isinstance(_intent, str) and _intent in cur_options_set

                assert len(cur_item) == 15
                cur_item = {
                    "id": str(cur_item["id"]).strip(),
                    "metadata": {
                        "id": str(cur_item["id"]).strip(),
                        "paper_year": int(cur_item["paper_year"]),
                        "original_task": str(cur_item["original_task"]).strip(),
                        "original_split": str(cur_item["original_split"]).strip(),
                        "text_form": str(cur_item["text_form"]).strip(),
                        "intent_type": str(cur_item["intent_type"]).strip(),
                        "is_synthetic": bool(cur_item["is_synthetic"]),
                        "is_sensitive": bool(cur_item["is_sensitive"]),
                        "domain_topic": list(cur_item["domain_topic"]),
                    },
                    "speaker": str(cur_item["speaker"]).strip(),
                    "context": str(cur_item["context"]).strip(),
                    "question": str(cur_item["question"]).strip(),
                    "options": cur_item["options"],
                    "answer_intent": cur_item["answer_intent"],
                    "answer_index": cur_item["answer_index"],
                }

                processed_items.append(cur_item)

            self.task_data[task_split]["data"] = processed_items

        if do_save:
            self.save_processed_data(self.unified_data_dir, verbose=True)
        return None

    def save_processed_data(
            self,
            save_data_dir: str,
            verbose: bool = False,
    ) -> None:
        # Save the unified dataset and metadata
        save_dir = os.path.join(save_data_dir, self.task_name)
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(self.task_meta, dict):
            meta_filepath = os.path.join(save_dir, f"metadata.json")
            DataIO.save_json(meta_filepath, self.task_meta, mode="w", indent=2, verbose=verbose)

        if isinstance(self.task_data, dict):
            for task_split in ["train", "valid", "test"]:
                data_filepath = os.path.join(save_dir, f"{task_split}.jsonl")
                assert isinstance(self.task_data[task_split]["data"], list)
                DataIO.save_jsonl(data_filepath, self.task_data[task_split]["data"], mode="w", verbose=verbose)


def main(
        cache_dir: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        seed: int = 42,
        verbose: bool = False,
        **kwargs
) -> None:
    """
    Evaluation Tasks and Datasets.

    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The directory of the project root.
    :param data_dir: The directory to load/store the data.
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setups
    logger = logger_setup("Tasks")
    random_setup(seed=seed, has_cuda=False)

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    from tasks.atis import TaskAtis
    from tasks.trec import TaskTrec
    from tasks.awc import TaskAWC

    from tasks.snips import TaskSnips
    from tasks.top import TaskTop
    from tasks.acl_cite import TaskAclCite

    from tasks.clinc import TaskClinc
    from tasks.facebook import TaskFacebook
    from tasks.twacs import TaskTwACS
    from tasks.mantis import TaskMantis
    from tasks.sci_cite import TaskSciCite

    from tasks.banking77 import TaskBanking77
    from tasks.slurp import TaskSlurp
    from tasks.acid import TaskACID
    from tasks.mcid import TaskMCID
    from tasks.mix_atis import TaskMixATIS
    from tasks.mix_snips import TaskMixSNIPS
    from tasks.empathetic_intents import TaskEmpatheticIntents
    from tasks.hint3 import TaskHint3
    from tasks.dstc8_sgd import TaskDSTC8SGD
    from tasks.multiwoz22 import TaskMultiWOZ22

    from tasks.multiwoz23 import TaskMultiWOZ23
    from tasks.hwu import TaskHWU
    from tasks.stanfordlu import TaskStanfordLU
    from tasks.mtop import TaskMTOP
    from tasks.xsid import TaskXSID
    from tasks.minds14 import TaskMinds14
    from tasks.conda import TaskConda
    from tasks.policyie import TaskPolicyIE
    from tasks.moral_stories import TaskMoralStories

    from tasks.nlupp import TaskNLUPP
    from tasks.plead import TaskPLEAD
    from tasks.iterater import TaskIterater
    from tasks.arxiv_edits import TaskArxivEdits

    from tasks.credit16 import TaskCredit16
    from tasks.vira import TaskVIRA
    from tasks.dstc11_t2 import TaskDSTC11T2

    from tasks.blendx import TaskBlendX
    from tasks.urs import TaskURS
    from tasks.intent_conan import TaskIntentConan
    from tasks.intention_qa import TaskIntentionQA
    from tasks.re3_sci2 import TaskRe3Sci2

    from tasks.ioinst import TaskIoInst
    from tasks.mathdial import TaskMathDial
    from tasks.dyndst import TaskDynDST
    from tasks.propa_gaze import TaskPropaGaze

    from tasks.recap import TaskRecap
    from tasks.malint import TaskMalInt
    from tasks.i2hate import TaskI2Hate

    # Dataset loading
    eval_tasks = TaskAtis(
        logger=logger, verbose=verbose, data_dir=data_dir,
        cache_dir=cache_dir, project_root_dir=project_root_dir)
    eval_tasks.load_task(do_save=True)

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
