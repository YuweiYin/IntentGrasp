# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskPLEAD(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "plead"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2022,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2022.tacl-1.82/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset (ACL publication - CC-BY)
            "intent_description": {},  # The description of each intent label
        }
        # Section 4: The Plead Dataset - Table 3
        #   To better assess the quality of the annotations, an expert annotator manually reviewed 50% in each category.

        self.task_data = {
            "train": {
                "filenames": ["dataset.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["dataset.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["dataset.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "not hate": "No clear toxicity targeting others.",  # 881
            "animosity": "To use implicit abusive language "
                         "targeting an individual or group based on their protected characteristics.",  # 497,
            "dehumanization": "To use dehumanizing comparisons "
                              "targeting an individual or group based on their protected characteristics.",  # 883,
            "derogation": "To use derogatory terms or insults "
                          "targeting an individual or group based on their protected characteristics.",  # 497,
            "pro hate crime": "To glorify, support, or deny "
                              "hateful actions, events, organizations, and individuals.",  # 171,
            "threatening": "To use threatening language "
                           "targeting an individual or group based on their protected characteristics.",  # 585,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "not hate": ["toxic speech", "abusive language"],  # 881
            "animosity": ["toxic speech", "abusive language"],  # 497,
            "dehumanization": ["toxic speech", "abusive language"],  # 883,
            "derogation": ["toxic speech", "abusive language"],  # 497,
            "pro hate crime": ["toxic speech", "abusive language"],  # 171,
            "threatening": ["toxic speech", "abusive language"],  # 585,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        split_qid_sets = {"train": set(), "valid": set(), "test": set()}
        for task_split in ["train", "valid", "test"]:
            idx_filepath = os.path.join(self.raw_data_dir, self.task_name, f"{task_split}_idx.csv")
            idx_file = DataIO.load_csv(str(idx_filepath), delimiter=",")
            assert isinstance(idx_file, list) and len(idx_file) > 0
            qid_list = [str(_idx[0]).strip() for _idx in idx_file]
            qid_set = set(qid_list)
            split_qid_sets[task_split] = qid_set

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            intent_raw = intent_raw.lower()
            if intent_raw == "nothate":
                intent_raw = "not hate"
            if intent_raw == "hatecrime":
                intent_raw = "pro hate crime"
            if intent_raw == "comparison":
                intent_raw = "dehumanization"
            return [intent_raw.strip()]

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
                assert file_format == "json"
                cur_data = DataIO.load_json(str(cur_filepath))
                assert isinstance(cur_data, dict) and "annotations" in cur_data
                cur_data_annotations = cur_data["annotations"]
                assert isinstance(cur_data_annotations, list) and len(cur_data_annotations) > 0
                cur_idx_set = split_qid_sets[task_split]
                # cur_data_annotations = [_d for _d in cur_data_annotations if str(_d["qid"]).strip() in cur_idx_set]

                cur_data_clear = []
                text_set = set()  # To avoid duplication
                for _d in cur_data_annotations:
                    assert isinstance(_d, dict)
                    if str(_d["qid"]).strip() not in cur_idx_set:
                        continue
                    if _d["copyid"] != 0:
                        continue
                    cur_text = str(_d["text"]).strip()
                    if cur_text in text_set:
                        continue
                    else:
                        text_set.add(cur_text)
                        cur_data_clear.append(_d)
                cur_split_raw_data += cur_data_clear

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item["rule"]).strip()
                for _cur_intent_label in _normalize_intent_label(cur_intent_label):
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
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "json"
                cur_data = DataIO.load_json(str(cur_filepath))
                assert isinstance(cur_data, dict) and "annotations" in cur_data
                cur_data_annotations = cur_data["annotations"]
                assert isinstance(cur_data_annotations, list) and len(cur_data_annotations) > 0
                cur_idx_set = split_qid_sets[task_split]

                cur_data_clear = []
                text_set = set()  # To avoid duplication
                for _d in cur_data_annotations:
                    assert isinstance(_d, dict)
                    if str(_d["qid"]).strip() not in cur_idx_set:
                        continue
                    if _d["copyid"] != 0:
                        continue
                    cur_text = str(_d["text"]).strip()
                    if cur_text in text_set:
                        continue
                    else:
                        text_set.add(cur_text)
                        cur_data_clear.append(_d)
                cur_file_raw_data = cur_data_clear

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item["text"]).strip()
                    cur_intent_label = str(raw_item["rule"]).strip()

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    assert len(_cur_intent_labels) == 1
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    if _cur_intent_labels[0] == "dehumanization":
                        cur_targets = list(raw_item["targets"])
                        assert len(cur_targets) == 1
                        cur_target = str(cur_targets[0])
                    elif _cur_intent_labels[0] == "not hate":
                        cur_targets = list(raw_item["targets"])
                        assert len(cur_targets) == 1
                        cur_target = str(cur_targets[0])
                    elif _cur_intent_labels[0] == "threatening":
                        cur_targets = list(raw_item["targets"])
                        assert len(cur_targets) == 1
                        cur_target = str(cur_targets[0])
                    elif _cur_intent_labels[0] == "derogation":
                        cur_targets = list(raw_item["targets"])
                        assert len(cur_targets) == 1
                        cur_target = str(cur_targets[0])
                    elif _cur_intent_labels[0] == "animosity":
                        cur_targets = list(raw_item["targets"])
                        assert len(cur_targets) == 1
                        cur_target = str(cur_targets[0])
                    elif _cur_intent_labels[0] == "pro hate crime":
                        assert "entity_span" in raw_item and "support_span" in raw_item
                        entity_span = str(raw_item["entity_span"]).strip()
                        cur_target = str(entity_span)
                    else:
                        raise ValueError(f"Unrecognized intent: {_cur_intent_labels[0]}")

                    cur_speaker = "poster"  # Twitter poster
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = (f"By targeting or mentioning \"{cur_target}\", "
                                       f"what is the intent of the {cur_speaker}?")
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
                        "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
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
