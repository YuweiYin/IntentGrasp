# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskSlurp(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "slurp"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2020.emnlp-main.588/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }

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
                "filenames": ["test.jsonl"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "calendar set": "To add events to the calendar.",  # 1142,
            "play music": "To play music.",  # 911,
            "weather query": "To check the weather.",  # 834,
            # "general quirky": "",  # 817,
            "calendar query": "To check the calendar.",  # 783,
            "qa factoid": "To ask about factoid.",  # 765,
            "news query": "To check the news.",  # 704,
            "email query": "To check the email.",  # 604,
            "email sendemail": "To send the email.",  # 523,
            "datetime query": "To check the datetime.",  # 490,
            "calendar remove": "To remove events from the calendar.",  # 419,
            # "social post": "To post social events.",  # 408,
            "play radio": "To play radio.",  # 394,
            "qa definition": "To ask about the definition.",  # 378,
            "cooking recipe": "To ask or talk about the cooking recipe.",  # 320,
            "transport query": "To check the transport.",  # 314,
            "lists query": "To check the lists.",  # 296,
            "play podcasts": "To play podcasts.",  # 282,
            "recommendation events": "To ask about event recommendations.",  # 258,
            "alarm set": "To set the alarm.",  # 253,
            "lists remove": "To remove items from the lists.",  # 251,
            "lists createoradd": "To create or add items to the lists.",  # 234,
            "recommendation locations": "To ask about location recommendation.",  # 234,
            "play audiobook": "To play audiobook.",  # 226,
            "music query": "To search for the music.",  # 215,
            "qa currency": "To ask about the currency.",  # 208,
            "iot hue lightoff": "To turn off the light.",  # 205,
            "qa stock": "To ask about the stock.",  # 202,
            "transport ticket": "To book a ticket or ask about related information.",  # 186,
            "alarm query": "To check the alarm.",  # 183,
            "iot hue lightchange": "To change the light color.",  # 183,
            "takeaway order": "To order takeaways.",  # 177,
            "takeaway query": "To check the takeaway information.",  # 176,
            "iot coffee": "To ask for some coffee.",  # 170,
            "email querycontact": "To check the email contact.",  # 168,
            "music likeness": "To express likeness to the music.",  # 164,
            "play game": "To play game.",  # 164,
            "audio volume mute": "To mute or turn off the audio.",  # 157,
            "transport traffic": "To ask about the traffic information.",  # 152,
            "transport taxi": "To book a taxi.",  # 150,
            "social query": "To check the news on social media.",  # 149,
            "audio volume up": "To turn up the audio volume.",  # 135,
            "iot cleaning": "To ask to do the cleaning.",  # 135,
            "qa maths": "To ask about math problems.",  # 116,
            "alarm remove": "To remove the alarm.",  # 113,
            "iot hue lightup": "To turn up the light.",  # 111,
            "iot hue lightdim": "To turn down the light.",  # 111,
            "recommendation movies": "To ask about movie recommendations.",  # 102,
            "general joke": "To ask for jokes.",  # 101,
            "datetime convert": "To convert the datetime.",  # 75,
            "iot wemo off": "To turn off the smart plug or socket.",  # 72,
            "audio volume down": "To turn down the audio volume.",  # 71,
            "email addcontact": "To add an email contact.",  # 70,
            # "query": "To make a general query.",  # 68,
            "music settings": "To change the music settings.",  # 64,
            "iot wemo on": "To turn on the smart plug or socket.",  # 64,
            "iot hue lighton": "To turn on the light.",  # 30,
            # "music": "",  # 27,
            # "general greet": "To greet.",  # 25,
            # "audio volume other": "",  # 23,
            "music dislikeness": "To express dislike of the music.",  # 20,
            # "quirky": "",  # 12,
            # "factoid": "",  # 10,
            # "sendemail": "",  # 9,
            # "set": "",  # 9,
            # "remove": "",  # 9,
            # "podcasts": "",  # 8,
            # "hue lightoff": "",  # 8,
            # "createoradd": "",  # 7,
            # "radio": "",  # 7,
            # "cooking query": "",  # 6,
            # "post": "",  # 6,
            # "joke": "",  # 5,
            # "currency": "",  # 5,
            # "game": "",  # 5,
            # "hue lightup": "",  # 4,
            # "coffee": "",  # 4,
            # "greet": "",  # 3,
            # "hue lightdim": "",  # 3,
            # "wemo off": "",  # 3,
            # "cleaning": "",  # 3,
            # "traffic": "",  # 2,
            # "querycontact": "",  # 1,
            # "addcontact": "",  # 1,
            # "ticket": "",  # 1,
            # "wemo on": "",  # 1,
            # "definition": "",  # 1,
            # "events": "",  # 1,
            # "convert": "",  # 1,
            # "settings": "",  # 1,
            # "volume other": "",  # 1,
            # "likeness": "",  # 1,
            # "locations": "",  # 1,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "music", "audio volume other", "quirky", "factoid", "sendemail",
            "set", "remove", "podcasts", "hue lightoff", "createoradd",
            "radio", "cooking query", "post", "joke", "currency",
            "game", "hue lightup", "coffee", "greet", "hue lightdim",
            "wemo off", "cleaning", "traffic", "querycontact", "addcontact",
            "ticket", "wemo on", "definition", "events", "convert",
            "settings", "volume other", "likeness", "locations", "general quirky",
            "social post", "general greet", "query",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            intent_clear = intent_raw.replace("_", " ")
            return [intent_clear.strip()]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        domain2intent = dict()
        intent2domain = dict()
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
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_text_raw = str(raw_item["sentence"]).strip()
                if len(cur_text_raw) == 0:
                    continue
                cur_topic_raw = str(raw_item["scenario"]).strip()
                cur_intent_label = str(raw_item["intent"]).strip()
                assert len(cur_topic_raw) > 0 and len(cur_intent_label) > 0
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                if len(_cur_intent_labels) == 0:
                    continue
                for _cur_intent_label in _cur_intent_labels:
                    if _cur_intent_label not in all_intents_dict:
                        all_intents_dict[_cur_intent_label] = 1
                    else:
                        all_intents_dict[_cur_intent_label] += 1

                    intent2domain[_cur_intent_label] = cur_topic_raw
                    if cur_topic_raw not in domain2intent:
                        domain2intent[cur_topic_raw] = [_cur_intent_label]
                    else:
                        domain2intent[cur_topic_raw].append(_cur_intent_label)

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
                assert file_format == "jsonl"
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item["sentence"]).strip()
                    if len(cur_text_raw) == 0:
                        continue
                    cur_intent_label = str(raw_item["intent"]).strip()
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
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in intent2domain
                        cur_domain, cur_topic = "daily life", intent2domain[_cur_intent_label]
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
