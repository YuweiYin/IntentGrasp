# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMixATIS(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "mix_atis"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2020.findings-emnlp.163/",  # URL of the dataset paper
            "license": "GPL",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }

        self.task_data = {
            "train": {
                "filenames": ["train.txt"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.txt"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.txt"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "air travel: flight": "To book flights or ask for general flight information.",  # 2235,
            "air travel: airfare": "To ask for airfare information.",  # 2170,
            "air travel: ground service": "To ask for ground service information.",  # 2148,
            "air travel: abbreviation": "To ask about abbreviations.",  # 2025,
            "air travel: aircraft": "To ask for aircraft information.",  # 1980,
            "air travel: airline": "To ask for airline information.",  # 1977,
            "air travel: flight time": "To ask about flight time.",  # 1953,
            "air travel: ground fare": "To ask for ground fare information.",  # 1904,
            "air travel: airport": "To ask for airport information.",  # 1886,
            "air travel: quantity": "To ask for the number of flights.",  # 1853,
            "air travel: distance": "To ask about flight distance.",  # 1848,
            "air travel: city": "To ask about city information or location.",  # 1844,
            "air travel: capacity": "To ask about flight capacity.",  # 1806,
            "air travel: flight no": "To ask for flight number.",  # 1740,
            "air travel: meal": "To ask for meal information.",  # 1726,
            "air travel: restriction": "To ask for flight restrictions.",  # 1610,
            "air travel: cheapest": "To ask for the cheapest flight.",  # 1316,
            "air travel: day name": "To ask for the day of the week.",  # 108,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _process_raw_txt(txt_raw: list) -> list:
            # Merge raw lines to construct instances
            assert isinstance(txt_raw, list) and len(txt_raw) > 0
            txt_raw_instances = []
            cur_text, cur_intents = "", ""
            for txt_line in txt_raw:
                txt_line = str(txt_line).strip()
                if len(txt_line) == 0:
                    # Done with one instance
                    txt_raw_instances.append([cur_text, cur_intents])
                    cur_text, cur_intents = "", ""  # reset the text and intents
                elif " " in txt_line:
                    # A token in the text
                    assert len(txt_line.split(" ")) == 2
                    cur_text += str(txt_line.split(" ")[0]).strip() + " "
                else:
                    # The intent of the text
                    cur_intents = txt_line
            return txt_raw_instances

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            if "#" in intent_raw:
                intents_raw = intent_raw.split("#")
            else:
                intents_raw = [intent_raw]

            intent_clear = []
            for _intent in intents_raw:
                _intent = _intent.strip()
                _intent_clear = ""
                for ch in _intent:
                    if ch.isupper():
                        _intent_clear += f" {ch.lower()}"
                    else:
                        _intent_clear += ch
                _intent_clear = _intent_clear.strip()
                _intent_clear = _intent_clear.replace("_", " ").strip()

                if _intent_clear.startswith("atis "):
                    _intent_clear = "air travel: " + _intent_clear[len("atis "):]

                intent_clear.append(_intent_clear)

            return intent_clear

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
                assert file_format == "txt"
                cur_data = DataIO.load_txt(str(cur_filepath))
                cur_data_instances = _process_raw_txt(cur_data)
                cur_split_raw_data += cur_data_instances

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                assert isinstance(raw_item, list) and len(raw_item) == 2
                cur_intent_label = str(raw_item[1]).strip()
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
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
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "txt"
                cur_data = DataIO.load_txt(str(cur_filepath))
                cur_data_instances = _process_raw_txt(cur_data)
                cur_file_raw_data = cur_data_instances

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    assert isinstance(raw_item, list) and len(raw_item) == 2
                    cur_text_raw = str(raw_item[0]).strip()
                    cur_intent_label = str(raw_item[1]).strip()
                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "user"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []
                    for _cur_intent_label in _cur_intent_labels:
                        cur_domain, cur_topic = "daily life", "air travel"
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
                        "paper_year": 2020,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "query",  # str: query/dialogue/monologue
                        "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
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
