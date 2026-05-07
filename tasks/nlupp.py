# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskNLUPP(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "nlupp"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2022,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2022.findings-naacl.154/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 3.4: Data Collection and Annotation
        #   Instead of relying on crowd-workers, 4 highly skilled annotators with dialogue and NLP expertise,
        #     also familiar with production environments, collected, annotated, and corrected the data.

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
                "filenames": ["banking", "hotels"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "request info": "To request information.",  # 557,
            "transfer payment deposit": "To transfer money, make payment, or deposit money.",  # 513,
            # "make open apply setup get activate": "",  # 381,
            "wrong: not working or not showing": "To report an issue: not working or not showing.",  # 342,
            "booking": "To inquire about booking.",  # 268,
            "card": "To inquire about the card.",  # 235,
            "account": "To inquire about the account.",  # 230,
            "change": "To make a change.",  # 204,
            "how much": "To ask how much something is.",  # 160,
            # "more or higher after": "",  # 153,
            "deny": "To deny.",  # 151,
            "affirm": "To affirm.",  # 150,
            "cancel close leave freeze": "To cancel, close, leave, or freeze the account or card.",  # 144,
            "fees interests": "To inquire about the fees or interests.",  # 104,
            "when": "To ask when something will happen.",  # 95,
            # "less or lower before": "",  # 93,
            "international": "To inquire about international usage or payments.",  # 90,
            "limits": "To inquire about the limits.",  # 90,
            # "do not know": "",  # 83,
            # "handoff": "",  # 83,
            "lost stolen": "To report or inquire about the lost card.",  # 81,
            "existing": "To inform of the existence of the account or card.",  # 81,
            "restaurant": "To inquire about restaurant booking.",  # 78,
            "why": "To ask for the reasons.",  # 77,
            "balance": "To inquire about the balance.",  # 71,
            "how": "To ask how to do something.",  # 71,
            "current": "To inquire about a current account.",  # 70,
            "thank": "To express gratitude.",  # 69,
            "overdraft": "To inquire about the overdraft.",  # 69,
            "savings": "To inquire about the savings.",  # 69,
            "loan": "To inquire about the loan.",  # 68,
            # "new": "",  # 67,
            "appointment": "To make an appointment or inquire about the appointment.",  # 65,
            "refund": "To ask for a refund or inquire about a refund.",  # 64,
            "contactless": "To inquire about the contactless card or service.",  # 63,
            "greet": "To express greetings.",  # 63,
            "mortgage": "To inquire about the mortgage.",  # 58,
            "debit": "To inquire about the debit card.",  # 58,
            "direct debit": "To inquire about the direct debit.",  # 54,
            "end call": "To end the call.",  # 53,
            # "spa": "",  # 52,
            # "acknowledge": "",  # 51,
            "withdrawal": "To inquire about money withdrawal.",  # 49,
            # "arrival": "",  # 44,
            "pin": "To talk about the PIN.",  # 43,
            "credit": "To inquire about the credit card.",  # 43,
            "room service": "To inquire about the room service.",  # 42,
            "cheque": "To inquire about the cheque account or deposit.",  # 39,
            "room amenities": "To inquire about the room amenities.",  # 39,
            "gym": "To inquire about the gym.",  # 36,
            "standing order": "To create or inquire about the standing order.",  # 34,
            "parking": "To inquire about parking.",  # 30,
            "business": "To inquire about the business account.",  # 29,
            "check in": "To check in.",  # 28,
            "how long": "To ask how long something will take.",  # 27,
            "swimming pool": "To inquire about the swimming pool.",  # 25,
            "wifi": "To inquire about WiFi.",  # 23,
            "check out": "To check out.",  # 22,
            "pets": "To inquire about pets.",  # 22,
            "repeat": "To ask for a repeat.",  # 21,
            "housekeeping": "To inquire about housekeeping.",  # 10,
            "accessibility": "To inquire about accessibility.",  # 10,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "make open apply setup get activate", "more or higher after", "less or lower before", "new", "spa",
            "do not know", "handoff", "acknowledge", "arrival",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            if intent_raw.isupper():
                return [intent_raw.lower()]

            intent_clear = ""
            for ch in intent_raw:
                if ch.isupper():
                    intent_clear += f" {ch.lower()}"
                else:
                    intent_clear += ch

            intent_clear = intent_clear.replace("_", " ").strip()
            if intent_clear == "wrong notworking notshowing":
                intent_clear = "wrong: not working or not showing"
            if intent_clear == "more higher after":
                intent_clear = "more or higher after"
            if intent_clear == "less lower before":
                intent_clear = "less or lower before"
            if intent_clear == "dont know":
                intent_clear = "do not know"
            if intent_clear == "accesibility":
                intent_clear = "accessibility"  # A typo error in the raw dataset
            if intent_clear == "room ammenities":
                intent_clear = "room amenities"  # A typo error in the raw dataset

            return [intent_clear.lower().strip()]

        intent_description = dict()
        domain2intents = dict()
        intent2domain = dict()
        ontology_filepath = os.path.join(self.raw_data_dir, self.task_name, "ontology.json")
        ontology_info = DataIO.load_json(str(ontology_filepath))
        assert isinstance(ontology_info, dict) and "intents" in ontology_info
        ontology_intents = ontology_info["intents"]
        assert isinstance(ontology_intents, dict)
        for intent_name, oi_v in ontology_intents.items():
            assert isinstance(intent_name, str) and isinstance(oi_v, dict)
            assert "description" in oi_v and "domain" in oi_v
            cur_description = str(oi_v["description"]).strip()
            cur_domain = str(list(oi_v["domain"])[0]).strip()

            if cur_domain == "hotels":
                cur_domain = "hotel"

            intent_name = str(intent_name).strip()
            intent_name = _normalize_intent_label(intent_name)[0]
            if intent_name in skip_intents:
                continue

            intent_description[intent_name] = cur_description
            intent2domain[intent_name] = cur_domain
            if cur_domain not in domain2intents:
                domain2intents[cur_domain] = [intent_name]
            else:
                domain2intents[cur_domain].append(intent_name)
        self.task_meta["intent_description"] = intent_description

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_dirname in cur_filenames:
                cur_dirpath = os.path.join(self.raw_data_dir, self.task_name, cur_dirname)
                cur_dirpath = str(cur_dirpath)
                assert os.path.isdir(cur_dirpath)
                cur_fn_list = os.listdir(cur_dirpath)
                cur_fn_list.sort()
                cur_fn_list = [_fn for _fn in cur_fn_list if _fn.startswith("fold") and _fn.endswith(".json")]
                for cur_filename in cur_fn_list:
                    file_format = cur_filename.split(".")[-1]
                    assert file_format == "json"
                    cur_filepath = os.path.join(cur_dirpath, cur_filename)
                    cur_data = DataIO.load_json(str(cur_filepath))
                    cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                if "intents" not in raw_item:
                    continue
                cur_intent_labels = list(raw_item["intents"])
                _cur_intent_labels = [_normalize_intent_label(_il)[0] for _il in cur_intent_labels]
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

        assert len(set(all_intents_dict.keys()) - set(self.task_meta["intent_description"].keys())) == 0
        assert len(set(self.task_meta["intent_description"].keys()) - set(all_intents_dict.keys())) == 0

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

            for cur_dirname in cur_filenames:
                cur_dirpath = os.path.join(self.raw_data_dir, self.task_name, cur_dirname)
                cur_dirpath = str(cur_dirpath)
                assert os.path.isdir(cur_dirpath)
                cur_fn_list = os.listdir(cur_dirpath)
                cur_fn_list.sort()
                cur_fn_list = [_fn for _fn in cur_fn_list if _fn.startswith("fold") and _fn.endswith(".json")]
                cur_file_raw_data = []
                for cur_filename in cur_fn_list:
                    file_format = cur_filename.split(".")[-1]
                    assert file_format == "json"
                    cur_filepath = os.path.join(cur_dirpath, cur_filename)
                    cur_data = DataIO.load_json(str(cur_filepath))
                    cur_file_raw_data += cur_data  # Note: each item is a multi-turn dialogue

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    if "intents" not in raw_item:
                        continue
                    cur_intent_labels = list(raw_item["intents"])
                    _cur_intent_labels = [_normalize_intent_label(_il)[0] for _il in cur_intent_labels]
                    assert len(_cur_intent_labels) > 0
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue

                    intent_raw_full = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1
                        assert _cur_intent_label in self.task_meta["intent_description"]
                        intent_raw_full.append(self.task_meta["intent_description"][_cur_intent_label].strip())

                    cur_text_raw = str(raw_item["text"]).strip()

                    cur_speaker = "user"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in intent2domain
                        cur_domain, cur_topic = intent2domain[_cur_intent_label], ""
                        if cur_domain == "hotel":
                            cur_domain, cur_topic = "daily life", "hotel"
                        if cur_domain == "banking":
                            cur_domain, cur_topic = "daily life", "banking"
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
