# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskStanfordLU(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "stanfordlu"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://ojs.aaai.org/index.php/AAAI/article/view/17541",  # URL of the dataset paper
            "license": "Open RAIL",  # the releasing license of the original dataset (AAAI publication: Open RAIL)
            "intent_description": {},  # The description of each intent label
        }

        self.task_data = {
            "train": {
                "filenames": ["stanford.0.spt_s_1", "stanford.0.spt_s_5",
                              "stanford.1.spt_s_1", "stanford.1.spt_s_5",
                              "stanford.2.spt_s_1", "stanford.2.spt_s_5"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["stanford.0.spt_s_1", "stanford.0.spt_s_5",
                              "stanford.1.spt_s_1", "stanford.1.spt_s_5",
                              "stanford.2.spt_s_1", "stanford.2.spt_s_5"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["stanford.0.spt_s_1", "stanford.0.spt_s_5",
                              "stanford.1.spt_s_1", "stanford.1.spt_s_5",
                              "stanford.2.spt_s_1", "stanford.2.spt_s_5"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            # "appreciate": "To express gratitude or appreciation.",  # 27248,
            # "query": "To ask a question.",  # 11077,
            # "inform": "To give information.",  # 10719,
            "request time": "To ask about the time.",  # 8564,
            "request route": "To ask about the route.",  # 8000,
            "confirm": "To confirm.",  # 5928,
            # "request poi": "To ask about POI.",  # 5541,
            "request date": "To ask about the date.",  # 5416,
            "request address": "To ask about the address.",  # 5138,
            "request weather": "To ask about the weather.",  # 5132,
            "navigate": "To navigate.",  # 4951,
            "remind": "To set a reminder.",  # 4251,
            "request party": "To ask about the party.",  # 4246,
            "schedule": "To schedule.",  # 4209,
            "request temperature": "To ask about the temperature.",  # 2546,
            "show in screen": "To ask for the information to be shown on the screen.",  # 2546,
            "request location": "To ask about the location.",  # 2331,
            "request traffic": "To ask about the traffic information.",  # 2284,
            "request agenda": "To ask about the agenda.",  # 2271,
            "request high temperature": "To ask about the high temperature.",  # 2123,
            # "request information": "To request information.",  # 2027,
            "command appointment": "To command an appointment.",  # 2002,
            "request low temperature": "To ask about the low temperature.",  # 1899,
            "list schedule": "To list the schedule.",  # 1852,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            # "appreciate": ["general", "greeting"],  # 27248,
            # "query": ["general", "query"],  # 11077,
            # "inform": ["general", "inform"],  # 10719,
            "request time": ["daily life", "schedule"],  # 8564,
            "request route": ["daily life", "navigation"],  # 8000,
            "confirm": ["general", "confirm"],  # 5928,
            # "request poi": ["", ""],  # 5541,
            "request date": ["daily life", "schedule"],  # 5416,
            "request address": ["daily life", "navigation"],  # 5138,
            "request weather": ["daily life", "weather"],  # 5132,
            "navigate": ["daily life", "navigation"],  # 4951,
            "remind": ["daily life", "schedule"],  # 4251,
            "request party": ["daily life", "schedule"],  # 4246,
            "schedule": ["daily life", "schedule"],  # 4209,
            "request temperature": ["daily life", "weather"],  # 2546,
            "show in screen": ["general", "clarification"],  # 2546,
            "request location": ["daily life", "navigation"],  # 2331,
            "request traffic": ["daily life", "navigation"],  # 2284,
            "request agenda": ["daily life", "schedule"],  # 2271,
            "request high temperature": ["daily life", "weather"],  # 2123,
            # "request information": ["general", "query"],  # 2027,
            "command appointment": ["daily life", "schedule"],  # 2002,
            "request low temperature": ["daily life", "weather"],  # 1899,
            "list schedule": ["daily life", "schedule"],  # 1852,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "appreciate", "query", "inform", "request poi", "request information",
        }

        def _normalize_intent_label(intent_raw: List[str]) -> List[str]:
            intent_clear = []
            for _i_raw in intent_raw:
                _i_raw = _i_raw.strip()
                _i_clear = ""
                for ch in _i_raw:
                    if ch.isupper():
                        _i_clear += f" {ch.lower()}"
                    else:
                        _i_clear += ch
                _i_clear = _i_clear.replace("_", " ").strip()
                intent_clear.append(_i_clear)

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
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename, f"{task_split}.json")
                cur_data = DataIO.load_json(str(cur_filepath))
                assert isinstance(cur_data, dict)
                cur_data_keys = list(cur_data.keys())
                assert len(cur_data_keys) == 1
                cur_topic_raw = cur_data_keys[0]
                cur_data_list = cur_data[cur_topic_raw]
                assert isinstance(cur_data_list, list)
                for _list in cur_data_list:
                    cur_split_raw_data += _list

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                assert isinstance(raw_item, dict) and "support" in raw_item and "query" in raw_item
                data_support = raw_item["support"]

                assert isinstance(data_support, dict) and "seq_ins" in data_support and "labels" in data_support
                assert (isinstance(data_support["seq_ins"], list) and isinstance(data_support["labels"], list) and
                        len(data_support["seq_ins"]) == len(data_support["labels"]))
                for seq_ins, labels in zip(data_support["seq_ins"], data_support["labels"]):
                    assert isinstance(seq_ins, list) and isinstance(labels, list)
                    cur_intent_labels = [str(_intent).strip() for _intent in labels]
                    _cur_intent_labels = _normalize_intent_label(cur_intent_labels)
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue
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
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename, f"{task_split}.json")
                cur_data = DataIO.load_json(str(cur_filepath))
                assert isinstance(cur_data, dict)
                cur_data_keys = list(cur_data.keys())
                assert len(cur_data_keys) == 1
                cur_topic_raw = cur_data_keys[0]
                cur_data_list = cur_data[cur_topic_raw]
                assert isinstance(cur_data_list, list)
                cur_file_raw_data = []
                for _list in cur_data_list:
                    cur_file_raw_data += _list

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    assert isinstance(raw_item, dict) and "support" in raw_item and "query" in raw_item
                    data_support = raw_item["support"]
                    # data_query = raw_item["query"]

                    assert isinstance(data_support, dict) and "seq_ins" in data_support and "labels" in data_support
                    assert (isinstance(data_support["seq_ins"], list) and isinstance(data_support["labels"], list) and
                            len(data_support["seq_ins"]) == len(data_support["labels"]))
                    for seq_ins, labels in zip(data_support["seq_ins"], data_support["labels"]):
                        assert isinstance(seq_ins, list) and isinstance(labels, list)
                        cur_text_raw = " ".join([str(_text).strip() for _text in seq_ins])
                        cur_intent_labels = [str(_intent).strip() for _intent in labels]
                        _cur_intent_labels = _normalize_intent_label(cur_intent_labels)
                        _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                        if len(_cur_intent_labels) == 0:
                            continue
                        for _cur_intent_label in _cur_intent_labels:
                            if _cur_intent_label not in all_intents_dict:
                                all_intents_dict[_cur_intent_label] = 1
                            else:
                                all_intents_dict[_cur_intent_label] += 1

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
                            "text_form": "query",  # str: query/dialogue/monologue
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
