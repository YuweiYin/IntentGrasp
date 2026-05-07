# -*- coding: utf-8 -*-

import os
# import re
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskSnips(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "snips"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2018,  # The year of publication/preprint
            "paper_url": "https://arxiv.org/abs/1805.10190",  # URL of the dataset paper
            "license": "CC0",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 5.1.1
        #   Crowdsourcing tasks were originally submitted to Amazon Mechanical Turk.
        #   A text query generation task consists in generating an example of user query matching a provided set
        #     of intent and slots.
        #   Each generated query goes through a validation process taking the form of a second crowdsourcing task,
        #     where at least two out of three new contributors must confirm its formulation, spelling, and intent.

        self.task_data = {
            "train": {
                "filenames": [
                    "train_SearchCreativeWork.json", "train_GetWeather.json", "train_BookRestaurant.json",
                    "train_PlayMusic.json", "train_AddToPlaylist.json", "train_RateBook.json",
                    "train_SearchScreeningEvent.json"
                ],
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
                "filenames": [
                    "test_SearchCreativeWork.json", "test_GetWeather.json", "test_BookRestaurant.json",
                    "test_PlayMusic.json", "test_AddToPlaylist.json", "test_RateBook.json",
                    "test_SearchScreeningEvent.json"
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "add to playlist": "To add music to the playlist.",  # 2042,
            "book restaurant": "To book a restaurant.",  # 2073,
            "get weather": "To get the weather information.",  # 2100,
            "play music": "To play music.",  # 2100,
            "rate book": "To rate a book.",  # 2056,
            "search creative work": "To search for a creative work.",  # 2054,
            "search screening event": "To search for screening events.",  # 2059,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "add to playlist": ["daily life", "music"],  # 2042,
            "book restaurant": ["daily life", "restaurant"],  # 2073,
            "get weather": ["daily life", "weather"],  # 2100,
            "play music": ["daily life", "music"],  # 2100,
            "rate book": ["daily life", "book"],  # 2056,
            "search creative work": ["daily life", "searching"],  # 2054,
            "search screening event": ["daily life", "searching"],  # 2059,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

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
            return [intent_clear.lower().strip()]

        all_intents_dict = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            # cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "json"
                cur_data = DataIO.load_json(str(cur_filepath), errors="ignore")  # errors="replace"

                # Record intent labels
                cur_file_raw_data = cur_data[list(cur_data.keys())[0]]
                cur_intent_label = str(cur_filename).replace(
                    "train_", "").replace(
                    "test_", "").replace(
                    ".json", "").strip()
                cur_intent_label = _normalize_intent_label(cur_intent_label)[0]
                if cur_intent_label not in all_intents_dict:
                    all_intents_dict[cur_intent_label] = len(cur_file_raw_data)
                else:
                    all_intents_dict[cur_intent_label] += len(cur_file_raw_data)

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
                cur_data = DataIO.load_json(str(cur_filepath), errors="ignore")  # errors="replace"
                cur_file_raw_data = cur_data[list(cur_data.keys())[0]]

                cur_intent_label = str(cur_filename).replace(
                    "train_", "").replace(
                    "test_", "").replace(
                    ".json", "").strip()
                cur_intent_label = _normalize_intent_label(cur_intent_label)[0]

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = " ".join([str(_t["text"]).strip() for _t in raw_item["data"]]).strip()
                    assert cur_intent_label in cur_split_intents
                    cur_split_intents[cur_intent_label] += 1

                    cur_speaker = "user"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    assert cur_intent_label in self.intent_label2statement
                    iu_answer_intent_raw.append(self.intent_label2statement[cur_intent_label])

                    cur_domain_topic = []
                    assert cur_intent_label in self.intent_label2category
                    cur_domain, cur_topic = self.intent_label2category[cur_intent_label]
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
                        "paper_year": 2018,  # The year of publication/preprint
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
