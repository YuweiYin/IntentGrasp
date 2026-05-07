# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskPolicyIE(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "policyie"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2021.acl-long.340/",  # URL of the dataset paper
            "license": "MIT",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 2.2: Data Annotation
        #   Post-editing and Quality Control. We compute an inter-annotator agreement for each
        #     annotated segment of policy documents using Krippendorff's Alpha.
        #   The annotators are asked to discuss their annotations and re-annotate those sections
        #     with token-level $\alpha_K$ falling below 0.75.
        #   After the re-annotation process, we calculate the agreement for the two categories of slots individually.
        #   The inter-annotator agreement is 0.87 and 0.84 for type-I and type-II slots, respectively.
        #   Then the adjudicators discuss and finalize the annotations. The adjudication process involves
        #     one of the annotators, the legal scholar, and the domain expert.

        self.task_data = {
            "train": {
                "filenames": ["train.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "data collection usage": "To set rules for data collection or usage.",  # 1105,
            "data security protection": "To set rules for data security or protection.",  # 132,
            "data sharing disclosure": "To set rules for data sharing or disclosure.",  # 463,
            "data storage retention deletion": "To set rules for data storage, retention, or deletion.",  # 293,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "data collection usage": ["policy making", ""],  # 1105,
            "data security protection": ["policy making", ""],  # 132,
            "data sharing disclosure": ["policy making", ""],  # 463,
            "data storage retention deletion": ["policy making", ""],  # 293,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        index_id2name = {
            0: "other",
            1: "data collection usage",
            2: "data security protection",
            3: "data sharing disclosure",
            4: "data storage retention deletion",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
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
                assert file_format == "parquet"
                cur_data = DataIO.load_parquet(str(cur_filepath))
                cur_data_list = []
                for _t, _i in zip(cur_data["text"].tolist(), cur_data["label"].tolist()):
                    _i = int(_i)
                    assert _i in index_id2name
                    if _i == 0:
                        continue  # skip "other" intent
                    cur_data_list.append({
                        "text": str(_t).strip(),
                        "intent": str(index_id2name[_i]).strip(),
                    })
                cur_split_raw_data += cur_data_list

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item["intent"]).strip()
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
                assert file_format == "parquet"
                cur_data = DataIO.load_parquet(str(cur_filepath))
                cur_data_list = []
                for _t, _i in zip(cur_data["text"].tolist(), cur_data["label"].tolist()):
                    _i = int(_i)
                    assert _i in index_id2name
                    if _i == 0:
                        continue  # skip "other" intent
                    cur_data_list.append({
                        "text": str(_t).strip(),
                        "intent": str(index_id2name[_i]).strip(),
                    })
                cur_file_raw_data = cur_data_list

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item["text"]).replace("\n", " ").strip()
                    cur_intent_label = str(raw_item["intent"]).strip()

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "policy"  # "policymaker"
                    iu_context = f"### Policy:\n{cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
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
                        "paper_year": 2021,  # The year of publication/preprint
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
