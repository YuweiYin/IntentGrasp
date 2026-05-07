# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskIterater(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "iterater"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2022,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2022.acl-long.250/",  # URL of the dataset paper
            "license": "Apache",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 4.2.3: Automatic Annotation
        #   Inter-Annotator Agreement. We measure inter-annotator agreement (IAA) using the Fleiss' \kappa.
        #   Table 6 shows the IAA across three domains. After the second round of re-annotation by proficient
        #     linguists, the Fleiss' \kappa increases to 0.5014, which indicates moderate agreement among annotators.

        self.task_data = {
            "train": {
                "filenames": ["train.jsonl"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.jsonl"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.jsonl"],  # "test_4k.jsonl"
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "clarity": "To make the text more formal, concise, readable, and understandable.",  # 147348,
            "coherence": "To make the text more cohesive, logically linked, and consistent as a whole.",  # 48591,
            "fluency": "To fix grammatical errors in the text.",  # 149256,
            "style": "To convey the writer's writing preferences, including emotions, tone, voice, etc.",  # 3236,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "clarity": ["writing", "edit"],  # 147348,
            "coherence": ["writing", "edit"],  # 48591,
            "fluency": ["writing", "edit"],  # 149256,
            "style": ["writing", "edit"],  # 3236,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            intent_clear = ""
            for ch in intent_raw:
                if ch.isupper():
                    intent_clear += f" {ch.lower()}"
                else:
                    intent_clear += ch

            return [intent_clear.strip()]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "jsonl"
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                cur_data = [_d for _d in cur_data if str(_d["before_sent"]).lower().strip() !=
                            str(_d["after_sent"]).lower().strip()]
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_context_raw = str(raw_item["before_sent"]).replace("\n", " ").strip()
                cur_text_raw = str(raw_item["after_sent"]).replace("\n", " ").strip()
                if len(cur_context_raw) == 0 or len(cur_text_raw) == 0 or cur_context_raw == cur_text_raw:
                    continue
                cur_intent_label = str(raw_item["labels"]).strip()
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                assert len(_cur_intent_labels) == 1
                for _cur_intent_label in _cur_intent_labels:
                    if _cur_intent_label not in all_intents_dict:
                        all_intents_dict[_cur_intent_label] = 1
                    else:
                        all_intents_dict[_cur_intent_label] += 1

        # Sort the intents dict by the count in descending order
        all_intents_tuple = list(all_intents_dict.items())
        all_intents_tuple.sort(key=lambda x: x[1], reverse=True)
        all_intents_list = [_tuple[0] for _tuple in all_intents_tuple]
        all_intents_dict = {_k: _v for _k, _v in all_intents_tuple}

        all_domains = dict()
        all_topics = dict()
        all_domain_topic = dict()

        # Then, obtain the context/query strings and intent labels (with counts) per split
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            # cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            cur_data_processed = []  # The processed data of the current split
            cur_split_intents = {_it: 0 for _it in all_intents_list}  # The intent counter of the current split
            item_idx = 0
            show_cnt = int(1e5)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "jsonl"
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                cur_data = [_d for _d in cur_data if str(_d["before_sent"]).lower().strip() !=
                            str(_d["after_sent"]).lower().strip()]
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_context_raw = str(raw_item["before_sent"]).replace("\n", " ").strip()
                    cur_text_raw = str(raw_item["after_sent"]).replace("\n", " ").strip()
                    if len(cur_context_raw) == 0 or len(cur_text_raw) == 0 or cur_context_raw == cur_text_raw:
                        continue
                    cur_intent_label = str(raw_item["labels"]).strip()
                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    assert len(_cur_intent_labels) == 1
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "writer"
                    iu_context = f"### Text Before Edit:\n{cur_context_raw}\n\n### Text After Edit:\n{cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker} for the edit?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2category
                        cur_domain, cur_topic = self.intent_label2category[_cur_intent_label]
                        cur_domain_topic.append([cur_domain, cur_topic])

                        if cur_domain not in all_domains:
                            all_domains[cur_domain] = 1
                        else:
                            all_domains[cur_domain] += 1

                        if cur_topic not in all_topics:
                            all_topics[cur_topic] = 1
                        else:
                            all_topics[cur_topic] += 1

                        cur_domain_topic_str = f"{cur_domain}---{cur_topic}"
                        if cur_domain_topic_str not in all_domain_topic:
                            all_domain_topic[cur_domain_topic_str] = 1
                        else:
                            all_domain_topic[cur_domain_topic_str] += 1

                    cur_id = f"{self.task_name}--{task_split}--{item_idx}"
                    item_idx += 1
                    cur_item_processed = {
                        # Metadata
                        "id": cur_id,  # str
                        "paper_year": 2022,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "monologue",  # str: query/dialogue/monologue
                        "intent_type": "single",  # "multiple" if multiple intents per item else "single"
                        "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
                        "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
                        "domain_topic": cur_domain_topic,  # List[List[str]]
                        "speaker": cur_speaker,  # str

                        # IntentGrasp instance
                        "context": iu_context,  # str
                        "question": iu_question_raw,  # str
                        "answer_intent": iu_answer_intent_raw,  # intent_statement: List[str]
                        "answer_index": [""],  # List[int] (could have multiple correct intents)
                        "options": ["", "", ""],  # List[str] (including the correct answer)
                    }
                    cur_data_processed.append(cur_item_processed)
                    if item_idx % show_cnt == 0:
                        self.logger.info(f">>> [{task_split}] Processed items: {item_idx}")

            # Limit the number of test instances
            if task_split == "test" and len(cur_data_processed) > self.max_num_test:
                cur_data_processed = random.sample(cur_data_processed, self.max_num_test)
            # Store the data
            self.task_data[task_split]["data"] = cur_data_processed
            self.task_data[task_split]["num_data"] = len(cur_data_processed)
            self.task_data[task_split]["intents"] = cur_split_intents

        self.task_meta["all_intents"] = all_intents_dict
        self.task_meta["num_intents"] = len(all_intents_dict)
        self.task_meta["all_domains"] = all_domains
        self.task_meta["all_topics"] = all_topics
        self.task_meta["all_domain_topic"] = all_domain_topic
        self.task_meta["intent_label2statement"] = self.intent_label2statement
        if do_save:
            self.save_processed_data(self.unified_data_dir, verbose=True)
        return None
