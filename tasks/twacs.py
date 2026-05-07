# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskTwACS(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "twacs"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2019,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/D19-1413/",  # URL of the dataset paper
            "license": "CC-BY-NC-SA",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 3: Data
        #   After investigating 500 randomly sampled conversations from TwACS, we established an annotation task with 14
        #     dialog intents and hired two annotators to label the sampled dialogs based on the user query utterances.
        #   The Cohen's kappa coefficient was 0.75, indicating a substantial agreement between the annotators.
        #   The disagreed items were resolved by a third annotator.

        self.task_data = {
            "train": {
                "filenames": [],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": [],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.csv"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "flight delay": "To comment on the flight delay.",  # 85,
            "baggage": "To comment on the baggage.",  # 40,
            "flight entertainment": "To comment on the flight entertainment.",  # 40,
            "terminal operation": "To comment on the terminal operation.",  # 34,
            "flight facility": "To comment on the flight facility.",  # 32,
            "flight staff": "To comment on the flight staff.",  # 30,
            "book flight": "To comment on the flight booking.",  # 27,
            "check in": "To comment on checking in.",  # 21,
            "customer service": "To comment on the customer service.",  # 19,
            "reward": "To comment on the reward.",  # 17,
            "change flight": "To comment on changing the flight.",  # 16,
            "terminal facility": "To comment on the terminal facility.",  # 13,
            "request feature": "To comment on the new features.",  # 10,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        intent_full_dict = {
            "BAGGAGE": "baggage",
            "BOOKFLT": "book flight",
            "CHGFLT": "change flight",
            "CHECKIN": "check in",
            "CUSTSUPP": "customer service",
            "DELAY": "flight delay",
            "INFLIGHTENT": "flight entertainment",
            "INFLIGHTFACS": "flight facility",
            "FLIGHTSTAFF": "flight staff",
            "NEWFEAT": "request feature",
            "REWARDS": "reward",
            "TERMFACS": "terminal facility",
            "TERMOPS": "terminal operation",
            "OTHER": "other",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            assert intent_raw in intent_full_dict
            res_intents = [intent_full_dict[intent_raw]]
            return res_intents

        def _clean_text(text_raw: str) -> str:
            text_raw = text_raw.strip()
            text_raw = text_raw.replace("__initials__", " ")
            text_raw = text_raw.replace("<rep__", "\n")
            text_raw = text_raw.replace("__rep>", " ")
            text_raw = text_raw.replace("<cust__", "\n")
            text_raw = text_raw.replace("__cust>", " ")
            text_raw = text_raw.replace("<company__", "\n")
            text_raw = text_raw.replace("__company>", " ")
            text_raw = text_raw.replace("__cust__", "company:")  # To customer:
            text_raw = text_raw.replace("__company__", "customer:")  # # To company:

            text_raw = text_raw.replace("\n\n", "\n")
            text_raw = "\n".join([_sent.strip() for _sent in text_raw.split("\n")])

            return text_raw.strip()

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
                assert file_format == "csv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = ","
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                cur_data = [_item for _item in cur_data if str(_item[1]).lower().strip() != "unk"]
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item[1]).strip()
                if cur_intent_label.lower() == "other":
                    continue
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
                assert file_format == "csv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = ","
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                cur_data = [_item for _item in cur_data if str(_item[1]).lower().strip() != "unk"]
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_intent_label = str(raw_item[1]).strip()
                    if cur_intent_label.lower() == "other":
                        continue
                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_context_raw = str(raw_item[4]).strip()
                    cur_text_raw = str(raw_item[3]).strip()

                    cur_context_clean = _clean_text(cur_context_raw)
                    cur_text_clean = _clean_text(cur_text_raw)

                    cur_speaker = "customer"
                    assert len(cur_context_clean) > 0
                    iu_context = f"### Chat History:\n{cur_context_clean}\n\n### Reply:\n{cur_text_clean}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}'s reply?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []  # "Airlines Customer Support"
                    for _cur_intent_label in _cur_intent_labels:
                        cur_domain, cur_topic = "customer support", "airline"
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
                        "paper_year": 2019,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "dialogue",  # str: query/dialogue/monologue
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
