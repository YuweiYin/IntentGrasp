# -*- coding: utf-8 -*-

import os
# import json
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMultiWOZ23(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "multiwoz23"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://link.springer.com/chapter/10.1007/978-3-030-88483-3_16",  # URL of the dataset paper
            "license": "MIT",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }

        self.task_data = {
            "train": {
                "filenames": ["train.csv"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.csv"],
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
            # "thank": "",  # 10505,
            "inform restaurant food": "To inform or inquire about restaurant food.",  # 5125,
            "inform train destination": "To inform or inquire about the train's destination.",  # 4312,
            "inform train day": "To inform or inquire about the date to take the train.",  # 4225,
            "inform train depart": "To inform or inquire about the train's departure.",  # 4059,
            "inform restaurant price": "To inform or inquire about the restaurant price.",  # 3879,
            "inform restaurant area": "To inform or inquire about the area information of the restaurant.",  # 3801,
            "inform hotel type": "To inform or inquire about the type of the hotel.",  # 3797,
            "inform hotel stay": "To inform or inquire about the stay in the hotel.",  # 3217,
            "inform restaurant time": "To inform or inquire about the time to dine in the restaurant.",  # 3126,
            "inform attraction type": "To inform or inquire about the type of the tourist attraction.",  # 3085,
            "inform hotel price": "To inform or inquire about the hotel price.",  # 2956,
            "inform hotel day": "To inform or inquire about the date to stay in the hotel.",  # 2923,
            "inform restaurant day": "To inform or inquire about the date to dine in the restaurant.",  # 2913,
            "inform hotel area": "To inform or inquire about the area information of the hotel.",  # 2834,
            "inform restaurant people": "To inform or inquire about people in the restaurant.",  # 2814,
            "inform hotel people": "To inform or inquire about people in the hotel.",  # 2760,
            "inform attraction area": "To inform or inquire about the area information of the tourist attraction.",  # 2628,
            "inform hotel stars": "To inform or inquire about hotel stars.",  # 2537,
            "inform train arrive": "To inform or inquire about the train's arrival.",  # 2446,
            "inform hotel parking": "To inform or inquire about parking near the hotel.",  # 2423,
            "inform hotel internet": "To inform or inquire about the internet in the hotel.",  # 2293,
            # "inform train leave": "To inform or inquire about the train's departure.",  # 2238,
            "inform train people": "To inform or inquire about people on the train.",  # 2224,
            "inform taxi depart": "To inform or inquire about the departure of the taxi.",  # 1569,
            # "bye": "",  # 1514,
            "inform restaurant name": "To inform or inquire about the restaurant name.",  # 1500,
            "request attraction post": "To request the tourist attraction post.",  # 1441,
            "inform taxi destination": "To inform or inquire about the destination of the taxi.",  # 1439,
            "request attraction phone": "To request the tourist attraction phone.",  # 1422,
            "request attraction address": "To request the tourist attraction address.",  # 1357,
            "inform hotel name": "To inform or inquire about the hotel name.",  # 1347,
            "request restaurant address": "To request the restaurant address.",  # 1265,
            "request restaurant phone": "To request the restaurant phone.",  # 1255,
            "request attraction fee": "To request the tourist attraction fee.",  # 1211,
            "inform attraction name": "To inform or inquire about the tourist attraction name.",  # 1116,
            "request train ticket": "To request a train ticket.",  # 1094,
            "inform taxi leave": "To inform or inquire about the departure of the taxi.",  # 1070,
            "request train reference": "To request the train reference number.",  # 1026,
            "inform restaurant": "To inform or inquire about the restaurant.",  # 912,
            "inform taxi": "To inform or inquire about the taxi.",  # 900,
            "request restaurant post": "To request the restaurant post.",  # 876,
            "request train time": "To request the travel time of the train.",  # 801,
            "request restaurant reference": "To request the restaurant reference number.",  # 793,
            "inform hotel": "To inform or inquire about the hotel.",  # 761,
            "request hotel reference": "To request the hotel reference number.",  # 673,
            "request hotel address": "To request the hotel address.",  # 629,
            "inform taxi arrive": "To inform or inquire about the arrival of the taxi.",  # 625,
            "inform train": "To inform or inquire about the train.",  # 553,
            "request hotel post": "To request the hotel post.",  # 547,
            "request hotel phone": "To request the hotel phone.",  # 542,
            "request train id": "To request the train ID.",  # 442,
            "request taxi car": "To request a taxi.",  # 426,
            "request attraction area": "To request the area information of the tourist attraction.",  # 403,
            "inform attraction": "To inform or inquire about the tourist attraction.",  # 381,
            # "request train leave": "To request the train's departure information.",  # 316,
            "request train arrive": "To request the arrival information of the train.",  # 315,
            "request attraction type": "To request the tourist attraction type.",  # 306,
            # "greet": "",  # 305,
            "request hotel price": "To request the hotel price.",  # 279,
            "request hotel internet": "To request internet in the hotel.",  # 243,
            "inform hospital": "To inform or inquire about the hospital.",  # 236,
            "request hotel parking": "To request the parking information near the hotel.",  # 234,
            "request hotel area": "To request the area information of the hotel.",  # 215,
            "request restaurant price": "To request the restaurant price.",  # 213,
            "inform police": "To inform or inquire about the police.",  # 187,
            "request restaurant food": "To request food at the restaurant.",  # 186,
            "request restaurant area": "To request the area information of the restaurant.",  # 176,
            "request hospital post": "To request the hospital post.",  # 153,
            "request hotel type": "To request the hotel type.",  # 144,
            "request hospital phone": "To request the hospital phone.",  # 130,
            "request police address": "To request the police address.",  # 112,
            "request taxi phone": "To request the taxi phone.",  # 105,
            "request police post": "To request the police post.",  # 104,
            "request hospital address": "To request the hospital address.",  # 104,
            "inform hospital department": "To inform or inquire about the hospital department.",  # 98,
            "inform police name": "To inform or inquire about the police name.",  # 72,
            "request hotel stars": "To request the hotel stars.",  # 69,
            "request police phone": "To request the police phone.",  # 59,
            "inform train id": "To inform or inquire about the train ID.",  # 21,
            # "welcome": "",  # 21,
            # "request booking day": "",  # 19,
            # "offer book train": "",  # 15,
            # "inform booking": "",  # 14,
            # "request booking people": "",  # 11,
            # "no offer restaurant": "",  # 10,
            "request attraction name": "To request the name of the tourist attraction.",  # 10,
            # "request booking stay": "",  # 8,
            # "request booking time": "",  # 6,
            # "no offer hotel": "",  # 5,
            # "request train day": "To request the day and time of the train.",  # 5,
            "inform attraction address": "To inform or inquire about the tourist attraction address.",  # 4,
            "request train people": "To request the people information on the train.",  # 4,
            "request restaurant name": "To request the restaurant name.",  # 4,
            "inform taxi car": "To inform or inquire about the taxi car.",  # 4,
            "inform train ticket": "To inform or inquire about the train ticket.",  # 3,
            "inform attraction choice": "To inform or inquire about the tourist attraction choice.",  # 3,
            "request train destination": "To request the train's destination.",  # 3,
            "inform attraction fee": "To inform or inquire about the tourist attraction fee.",  # 3,
            "inform restaurant choice": "To inform or inquire about the restaurant choice.",  # 3,
            "inform hotel choice": "To inform or inquire about the hotel choice.",  # 3,
            # "no offer attraction": "",  # 2,
            "inform hospital phone": "To inform or inquire about the hospital phone.",  # 2,
            # "no offer train": "",  # 2,
            "request taxi depart": "To request the taxi's departure information.",  # 2,
            "request taxi destination": "To request the taxi's destination.",  # 2,
            "recommend attraction name": "To recommend a tourist attraction.",  # 2,
            "request taxi arrive": "To request the taxi's arrival information.",  # 2,
            # "inform booking day": "",  # 2,
            # "book booking time": "",  # 1,
            "request hospital department": "To request the hospital department.",  # 1,
            "inform hospital post": "To inform or inquire about the hospital post.",  # 1,
            "recommend attraction post": "To recommend a tourist attraction.",  # 1,
            "inform restaurant address": "To inform or inquire about the restaurant address.",  # 1,
            "select hotel": "To select a hotel.",  # 1,
            "inform police phone": "To inform or inquire about the police phone.",  # 1,
            "inform police address": "To inform or inquire about the police address.",  # 1,
            # "book booking reference": "",  # 1,
            "select restaurant": "To select a restaurant.",  # 1,
            "select attraction fee": "To select the fee of a tourist attraction.",  # 1,
            "recommend hotel name": "To recommend a hotel.",  # 1,
            # "request taxi leave": "To request the taxi's departure information.",  # 1,
            # "book booking": "",  # 1,
            "inform hotel address": "To inform or inquire about the hotel address.",  # 1,
            "request train depart": "To request the train's departure information.",  # 1,
            "request hotel name": "To request the hotel name.",  # 1,
            # "offer booked train reference": "",  # 1,
            # "no offer train depart": "",  # 1,
            # "inform booking people": "",  # 1,
            # "offer booked train arrive": "",  # 1,
            # "offer booked train destination": "",  # 1,
            # "offer booked train id": "",  # 1,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "inform restaurant food": ["daily life", "restaurant"],
            "inform restaurant price": ["daily life", "restaurant"],
            "inform restaurant area": ["daily life", "restaurant"],
            "inform restaurant time": ["daily life", "restaurant"],
            "inform restaurant day": ["daily life", "restaurant"],
            "inform restaurant people": ["daily life", "restaurant"],
            "inform restaurant name": ["daily life", "restaurant"],
            "request restaurant address": ["daily life", "restaurant"],
            "request restaurant phone": ["daily life", "restaurant"],
            "inform restaurant": ["daily life", "restaurant"],
            "request restaurant post": ["daily life", "restaurant"],
            "request restaurant reference": ["daily life", "restaurant"],
            "request restaurant price": ["daily life", "restaurant"],
            "request restaurant food": ["daily life", "restaurant"],
            "request restaurant area": ["daily life", "restaurant"],
            "no offer restaurant": ["daily life", "restaurant"],
            "request restaurant name": ["daily life", "restaurant"],
            "inform restaurant choice": ["daily life", "restaurant"],
            "inform restaurant address": ["daily life", "restaurant"],
            "select restaurant": ["daily life", "restaurant"],

            "inform attraction type": ["daily life", "tourist attraction"],
            "inform attraction area": ["daily life", "tourist attraction"],
            "request attraction post": ["daily life", "tourist attraction"],
            "request attraction phone": ["daily life", "tourist attraction"],
            "request attraction address": ["daily life", "tourist attraction"],
            "request attraction fee": ["daily life", "tourist attraction"],
            "inform attraction name": ["daily life", "tourist attraction"],
            "request attraction area": ["daily life", "tourist attraction"],
            "inform attraction": ["daily life", "tourist attraction"],
            "request attraction type": ["daily life", "tourist attraction"],
            "request attraction name": ["daily life", "tourist attraction"],
            "inform attraction address": ["daily life", "tourist attraction"],
            "inform attraction choice": ["daily life", "tourist attraction"],
            "inform attraction fee": ["daily life", "tourist attraction"],
            "no offer attraction": ["daily life", "tourist attraction"],
            "recommend attraction name": ["daily life", "tourist attraction"],
            "recommend attraction post": ["daily life", "tourist attraction"],
            "select attraction fee": ["daily life", "tourist attraction"],

            "inform hotel type": ["daily life", "hotel"],
            "inform hotel stay": ["daily life", "hotel"],
            "inform hotel price": ["daily life", "hotel"],
            "inform hotel day": ["daily life", "hotel"],
            "inform hotel area": ["daily life", "hotel"],
            "inform hotel people": ["daily life", "hotel"],
            "inform hotel stars": ["daily life", "hotel"],
            "inform hotel parking": ["daily life", "hotel"],
            "inform hotel internet": ["daily life", "hotel"],
            "inform hotel name": ["daily life", "hotel"],
            "inform hotel": ["daily life", "hotel"],
            "request hotel reference": ["daily life", "hotel"],
            "request hotel address": ["daily life", "hotel"],
            "request hotel post": ["daily life", "hotel"],
            "request hotel phone": ["daily life", "hotel"],
            "request hotel price": ["daily life", "hotel"],
            "request hotel internet": ["daily life", "hotel"],
            "request hotel parking": ["daily life", "hotel"],
            "request hotel area": ["daily life", "hotel"],
            "request hotel type": ["daily life", "hotel"],
            "request hotel stars": ["daily life", "hotel"],
            "no offer hotel": ["daily life", "hotel"],
            "inform hotel choice": ["daily life", "hotel"],
            "select hotel": ["daily life", "hotel"],
            "recommend hotel name": ["daily life", "hotel"],
            "inform hotel address": ["daily life", "hotel"],
            "request hotel name": ["daily life", "hotel"],

            "inform taxi depart": ["daily life", "taxi"],
            "inform taxi destination": ["daily life", "taxi"],
            "inform taxi leave": ["daily life", "taxi"],
            "inform taxi": ["daily life", "taxi"],
            "inform taxi arrive": ["daily life", "taxi"],
            "request taxi car": ["daily life", "taxi"],
            "request taxi phone": ["daily life", "taxi"],
            "inform taxi car": ["daily life", "taxi"],
            "request taxi depart": ["daily life", "taxi"],
            "request taxi destination": ["daily life", "taxi"],
            "request taxi arrive": ["daily life", "taxi"],
            "request taxi leave": ["daily life", "taxi"],

            "inform train destination": ["daily life", "train"],
            "inform train day": ["daily life", "train"],
            "inform train depart": ["daily life", "train"],
            "inform train arrive": ["daily life", "train"],
            # "inform train leave": ["daily life", "train"],
            "inform train people": ["daily life", "train"],
            "request train ticket": ["daily life", "train"],
            "request train reference": ["daily life", "train"],
            "request train time": ["daily life", "train"],
            "inform train": ["daily life", "train"],
            "request train id": ["daily life", "train"],
            "request train leave": ["daily life", "train"],
            "request train arrive": ["daily life", "train"],
            "inform train id": ["daily life", "train"],
            "offer book train": ["daily life", "train"],
            "request train day": ["daily life", "train"],
            "request train people": ["daily life", "train"],
            "inform train ticket": ["daily life", "train"],
            "request train destination": ["daily life", "train"],
            "no offer train": ["daily life", "train"],
            "request train depart": ["daily life", "train"],
            "offer booked train reference": ["daily life", "train"],
            "no offer train depart": ["daily life", "train"],
            "offer booked train arrive": ["daily life", "train"],
            "offer booked train destination": ["daily life", "train"],
            "offer booked train id": ["daily life", "train"],

            "inform hospital": ["daily life", "hospital"],
            "request hospital post": ["daily life", "hospital"],
            "request hospital phone": ["daily life", "hospital"],
            "request hospital address": ["daily life", "hospital"],
            "inform hospital department": ["daily life", "hospital"],
            "inform hospital phone": ["daily life", "hospital"],
            "request hospital department": ["daily life", "hospital"],
            "inform hospital post": ["daily life", "hospital"],

            "inform police": ["daily life", "police"],
            "request police address": ["daily life", "police"],
            "request police post": ["daily life", "police"],
            "inform police name": ["daily life", "police"],
            "request police phone": ["daily life", "police"],
            "inform police phone": ["daily life", "police"],
            "inform police address": ["daily life", "police"],
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            # Booking
            "request booking day", "inform booking", "request booking people", "request booking stay",
            "request booking time", "inform booking day", "book booking time", "book booking reference",
            "book booking", "inform booking people",
            # Greeting
            "thank", "greet", "welcome", "bye",
            # other
            "request train day",
            "offer book train", "offer booked train arrive", "offer booked train destination",
            "offer booked train reference", "offer booked train id",
            "no offer train", "no offer train depart", "no offer restaurant", "no offer hotel", "no offer attraction",
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
            intent_clear = intent_clear.replace("-", " ").strip()
            intent_clear = intent_clear.replace("  ", " ").strip()
            intent_clear = intent_clear.replace("addr", "address").strip()
            intent_clear = intent_clear.replace("dest", "destination").strip()
            if intent_clear.endswith("ref"):
                intent_clear = intent_clear.rstrip("ref").strip() + " reference"
            intent_clear = intent_clear.replace("general none", "").strip()
            if intent_clear.endswith("none"):
                intent_clear = intent_clear.rstrip("none").strip()

            intent_clear = intent_clear.lower().strip()
            if intent_clear == "inform train leave":
                intent_clear = "inform train depart"
            if intent_clear == "request train leave":
                intent_clear = "request train depart"
            if intent_clear == "request taxi leave":
                intent_clear = "request taxi depart"

            return [intent_clear]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        # all_intent2text = dict()
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
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_labels = str(raw_item[4]).strip()
                if "#" in cur_intent_labels:
                    _cur_intent_labels = [_normalize_intent_label(_il)[0] for _il in cur_intent_labels.split("#")]
                else:
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
        too_many_cnt = 0
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
                    cur_intent_labels = str(raw_item[4]).strip()
                    if "#" in cur_intent_labels:
                        _cur_intent_labels = [_normalize_intent_label(_il)[0] for _il in cur_intent_labels.split("#")]
                    else:
                        _cur_intent_labels = _normalize_intent_label(cur_intent_labels)
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue

                    cur_concat_context = str(raw_item[5]).strip()
                    assert "[SEP]" in cur_concat_context
                    cur_concat_context_list = cur_concat_context.split("[SEP]")
                    assert len(cur_concat_context_list) == 2
                    cur_text_raw = cur_concat_context_list[-1].strip()
                    cur_context = cur_concat_context_list[0].strip()

                    cur_speaker = "user"
                    if len(cur_context) > 0:
                        iu_context = f"### Chat History:\n{cur_context}\n\n{cur_speaker}: {cur_text_raw}"
                    else:
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
                        "text_form": "dialogue",  # str: query/dialogue/monologue
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
                    if len(iu_answer_intent_raw) >= 10:
                        too_many_cnt += 1
                        continue
                    cur_data_processed.append(cur_item_processed)

            # Limit the number of test instances
            if task_split == "test" and len(cur_data_processed) > self.max_num_test:
                cur_data_processed = random.sample(cur_data_processed, self.max_num_test)
            # Store the data
            self.task_data[task_split]["data"] = cur_data_processed
            self.task_data[task_split]["num_data"] = len(cur_data_processed)
            self.task_data[task_split]["intents"] = cur_split_intents

        # if self.verbose:
        #     self.logger.info(f">>> !!! >>> too_many_cnt = {too_many_cnt}")   # too_many_cnt = 1
        self.task_meta["all_intents"] = all_intents_dict
        self.task_meta["num_intents"] = len(all_intents_dict)
        self.task_meta["all_domains"] = all_domains
        self.task_meta["all_topics"] = all_topics
        self.task_meta["all_domain_topic"] = all_domain_topic
        self.task_meta["intent_label2statement"] = self.intent_label2statement
        if do_save:
            self.save_processed_data(self.unified_data_dir, verbose=True)
        return None
