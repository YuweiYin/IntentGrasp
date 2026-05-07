# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskCredit16(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "credit16"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2023,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2023.findings-emnlp.636/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset (ACL publication - CC-BY)
            "intent_description": {},  # The description of each intent label
        }
        # Section 4: Experiment
        #   Also, we pre-process the CREDIT16 dataset by lower-casing and then removing punctuation characters
        #     and emojis from the utterances and intent labels. Next, we remove new-line characters and
        #     repeated white space characters from each utterance sample.
        # Appendix: More details on the CREDIT16 dataset
        #   For pre-selection of which samples would be annotated by humans, we first restricted to which
        #     utterances would receive at least two labels under our automated process above. (We note that
        #     not all utterances retained more than  1 label by the end of our rounds of annotation.)
        #   Next, we split the dataset to be annotated by 3 linguists contracted to our company
        #     for a first round of annotations.

        self.task_data = {
            "train": {
                "filenames": ["train.csv"],
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
            # "greeting": "To express greetings.",  # 333,
            "dnc": "To ask the other side not to contact me anymore.",  # 325, (dnc = do not contact)
            # "not interested": "",  # 309,
            "cost too high": "To express that the cost is too high.",  # 242,
            # "specific equipment cost": "To inquire about the cost of specific equipment.",  # 214,
            # "transfer success": "",  # 211,
            "what does it cost": "To ask what it costs.",  # 185,
            "call": "To call now or schedule a call.",  # 185 + 33 = 218
            # "schedule call day n time": "To schedule a call.",  # 185,
            "is interested": "To express interest.",  # 180,
            # "courtesy or statement": "",  # 175,
            # "cost demand": "",  # 153,
            # "chose competitor": "",  # 148,
            "future interest": "To express the future interest.",  # 129,
            # "confirmation": "To confirm.",  # 102,
            # "apology": "To apologize.",  # 101,
            # "needs income": "",  # 100,
            # "existing system": "",  # 87,
            # "promo info": "To inquire about the promo information.",  # 83,
            # "delay work": "",  # 77,
            # "delay new house": "",  # 74,
            # "home specifications": "To talk about home specifications.",  # 71,
            # "delay evaluating": "",  # 69,
            "excessive contact": "To complain about the excessive contacts.",  # 66,
            "send quote": "To ask for the quote.",  # 64,
            # "delay conditional": "",  # 60,
            # "delay will reply": "",  # 53,
            # "new home": "To move to a new home.",  # 51,
            # "credit denied": "",  # 48,
            "text only": "To ask for text-only contact.",  # 46,
            "free install": "To inquire about free installation.",  # 41,
            "wrong number": "To report an issue: wrong name.",  # 39,
            "in contract": "To inform of the current contract.",  # 35,
            "agent contact info": "To ask for the contact information.",  # 35,
            "about contract": "To inquire about the contract.",  # 33,
            # "call now": "To call now or schedule a call.",  # 33,
            # "has product": "",  # 33,
            "how does it work": "To ask how something works.",  # 30,
            "send info": "To ask the other side to send related information or documents.",  # 30,
            # "delay finances": "",  # 29,
            # "delay hours": "",  # 29,
            # "delay funeral": "",  # 27,
            # "buy equipment": "To talk or inquire about equipment purchase.",  # 27,
            # "delay family sick": "",  # 27,
            # "negation": "To express negation.",  # 26,
            "system error": "To report a system error.",  # 24,
            # "no monitoring": "",  # 24,
            # "equipment compatible": "To talk or inquire about equipment compatibility.",  # 23,
            "not eligible": "To report an issue: not eligible.",  # 22,
            "negative call center experience": "To report a negative experience with the call center.",  # 21,
            # "delay sick": "",  # 20,
            # "misunderstood": "",  # 20,
            # "info for someone else": "",  # 19,
        }  # intent labels --> intent statements
        # Note: skip the "home security" domain, and filter out inappropriate intents.

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "greeting", "not interested", "transfer success", "courtesy or statement", "cost demand",
            "chose competitor", "needs income", "existing system", "promo info", "delay work",
            "delay new house", "delay evaluating", "delay conditional", "delay will reply", "delay finances",
            "delay hours", "delay funeral", "delay family sick", "delay sick", "new home",
            "credit denied", "has product", "no monitoring", "buy equipment", "equipment compatible",
            "specific equipment cost", "misunderstood", "info for someone else", "confirmation", "negation",
            "apology", "home specifications",
        }  # Note: some labels are not informative or even not appropriate to the text

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            if "," in intent_raw:
                intent_clear = []
                for _intent in intent_raw.split(","):
                    _intent = _intent.strip()
                    _intent = _intent.replace("_", " ").strip()
                    if _intent in ["call now", "schedule call day n time"]:
                        _intent = "call"
                    intent_clear.append(_intent)
                return intent_clear
            else:
                return [intent_raw.strip()]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        all_intent2text = dict()
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
                # csv_header = cur_data[0]
                cur_data = cur_data[1:]  # ignore the csv header
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item[2]).strip()
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                if len(_cur_intent_labels) == 0:
                    continue
                for _cur_intent_label in _cur_intent_labels:
                    if _cur_intent_label not in all_intents_dict:
                        all_intents_dict[_cur_intent_label] = 1
                    else:
                        all_intents_dict[_cur_intent_label] += 1

                    cur_text_raw = str(raw_item[1]).strip()
                    if _cur_intent_label not in all_intent2text:
                        all_intent2text[_cur_intent_label] = [cur_text_raw]
                    else:
                        all_intent2text[_cur_intent_label].append(cur_text_raw)

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
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item[1]).strip()
                    cur_intent_label = str(raw_item[2]).strip()

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue
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
                        "paper_year": 2023,  # The year of publication/preprint
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
