# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMTOP(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "mtop"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2021.eacl-main.257/",  # URL of the dataset paper
            "license": "CC-BY-SA",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 3.1: Dataset Creation
        #   We ask crowdsourced workers to generate natural language sentences that they would ask a system
        #     which could assist in queries corresponding to our chosen domains. These queries are labeled
        #     by two annotators. A third annotator is used only to adjudicate any disagreements.
        #   Post-editing and Quality Control: We further run two rounds of quality control over translated utterances
        #     and slots, and revise the data accordingly.
        #   83% of the data was marked as good quality data and passed our quality standards,
        #     which can be interpreted as the inter-annotator agreement rate on the translated data.
        #   Based on this feedback, we remove low quality annotations from the dataset.

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
            "get weather": "To get the weather information.",  # 2312,
            "create reminder": "To set a reminder.",  # 1716,
            "create call": "To make a call.",  # 1716,
            "get stories news": "To get stories or news.",  # 1554,
            "create alarm": "To set an alarm.",  # 1484,
            "send message": "To send a message.",  # 1326,
            "play music": "To play music.",  # 1252,
            "get info recipes": "To get the recipe details.",  # 1111,
            "get event": "To get event information.",  # 972,
            # "get contact": "To get the contact information.",  # 914,
            "get recipes": "To obtain the recipes.",  # 613,
            "create timer": "To set a timer.",  # 602,
            "delete reminder": "To delete the reminder.",  # 513,
            "get reminder": "To check the reminder.",  # 488,
            "get message": "To check the message.",  # 321,
            # "update call": "",  # 311,
            "get availability": "To get the availability.",  # 294,
            "get timer": "To check the timer.",  # 282,
            "get location": "To check the location.",  # 251,
            # "question news": "",  # 192,
            "update reminder date time": "To update the datetime of the reminder.",  # 191,
            "add time timer": "To add time to the timer.",  # 169,
            "end call": "To end the call.",  # 168,
            "get alarm": "To check the alarm.",  # 154,
            "get employer": "To ask the employer information.",  # 149,
            # "get info contact": "To get the contact information.",  # 147,
            "is true recipes": "To check the ingredients of the recipe.",  # 121,
            "snooze alarm": "To snooze the alarm.",  # 118,
            "pause timer": "To pause the timer.",  # 111,
            "get track info music": "To get information about the music track.",  # 107,
            "set unavailable": "To set the status as unavailable.",  # 100,
            "delete alarm": "To delete the alarm.",  # 89,
            "silence alarm": "To set off the alarm.",  # 86,
            "delete timer": "To delete the timer.",  # 80,
            "get call": "To check the calls.",  # 79,
            "set available": "To set the status as available.",  # 76,
            "get employment time": "To ask about the employment time.",  # 75,
            "update alarm": "To update the alarm.",  # 75,
            "get age": "To get the age information.",  # 74,
            "ignore call": "To ignore the call.",  # 73,
            "switch call": "To switch the call.",  # 73,
            "replay music": "To replay the music.",  # 72,
            "get education time": "To get the education time.",  # 70,
            "skip track music": "To skip the current music track.",  # 70,
            "update timer": "To update the timer.",  # 68,
            "set default provider music": "To set the default music provider.",  # 66,
            "answer call": "To answer the call.",  # 62,
            "subtract time timer": "To subtract time from the timer.",  # 62,
            "set rsvp no": "To reply no for RSVP.",  # 62,
            "update method call": "To change the calling method.",  # 60,
            "resume timer": "To resume the timer.",  # 59,
            "set rsvp interested": "To reply to indicate interest for RSVP.",  # 58,
            "get call time": "To check the calling time.",  # 56,
            "get reminder date time": "To get the datetime of the reminder.",  # 56,
            "restart timer": "To restart the timer.",  # 55,
            "start shuffle music": "To shuffle the music.",  # 48,
            "previous track music": "To play the previous music track.",  # 48,
            "set rsvp yes": "To reply yes for RSVP.",  # 45,
            "create playlist music": "To create a music playlist.",  # 40,
            # "question music": "",  # 40,
            "get sunset": "To get the sunset information.",  # 33,
            "update reminder todo": "To add an item to the reminder.",  # 29,
            "remove from playlist music": "To remove a song from the music playlist.",  # 27,
            # "get message contact": "To check the message contact.",  # 25,
            "get sunrise": "To get the sunrise information.",  # 24,
            "share event": "To share events.",  # 24,
            "get attendee event": "To get the event attendee.",  # 24,
            # "get call contact": "To get the call contact.",  # 23,
            "get date time event": "To get the event datetime.",  # 20,
            "stop music": "To stop playing the music.",  # 20,
            # "get contact method": "",  # 20,
            "pause music": "To pause the music.",  # 19,
            "get category event": "To get the category of the event.",  # 19,
            "update reminder": "To update the reminder.",  # 19,
            "get reminder amount": "To get the number of reminders.",  # 18,
            "get mutual friends": "To get mutual friends.",  # 17,
            # "get undergrad": "",  # 17,
            "get reminder location": "To ask for the location of the reminder.",  # 17,
            "hold call": "To hold the call.",  # 16,
            "get details news": "To get details of the news.",  # 16,
            "get education degree": "To get the education degree information.",  # 15,
            "add to playlist music": "To add a song to a music playlist.",  # 14,
            "loop music": "To loop the song.",  # 14,
            "play media": "To play media.",  # 12,
            "like music": "To like the song.",  # 12,
            "rewind music": "To rewind the song.",  # 12,
            "dislike music": "To dislike the song.",  # 12,
            "fast forward music": "To fast-forward the song.",  # 10,
            "set default provider calling": "To set the default calling provider.",  # 9,
            "cancel message": "To cancel a message.",  # 9,
            "get life event": "To get life events about a person.",  # 8,
            "get job": "To get a job.",  # 8,
            # "get major": "",  # 8,
            "delete playlist music": "To delete a song from the music playlist.",  # 8,
            "get life event time": "To get the time of life events.",  # 8,
            "resume call": "To resume calling.",  # 7,
            "prefer": "To express preference.",  # 6,
            "unloop music": "To unloop the song.",  # 6,
            "resume music": "To resume playing the song.",  # 6,
            # "help reminder": "",  # 6,
            "update reminder location": "To update the location of the reminder.",  # 5,
            "repeat all off music": "To stop repeating all music.",  # 5,
            "get lyrics music": "To get the lyrics.",  # 5,
            "merge call": "To merge calls.",  # 3,
            # "disprefer": "",  # 3,
            "get air quality": "To get the air quality.",  # 3,
            "cancel call": "To cancel the call.",  # 3,
            "repeat all music": "To repeat playing all songs.",  # 2,
            # "get language": "To get the language information.",  # 2,
            "stop shuffle music": "To stop shuffling the music.",  # 1,
            # "follow music": "To follow the music.",  # 1,
            # "get gender": "To get the gender information.",  # 1,
            # "get group": "To get the group information.",  # 1,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "update call", "question news", "question music", "get undergrad", "get major",
            "get message contact", "get call contact", "get contact", "get info contact", "get contact method",
            "disprefer", "follow music", "get language", "get gender", "get group",
            "help reminder",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            assert intent_raw.startswith("IN:")
            intent_raw = intent_raw[len("IN:"):].strip()
            intent_raw = intent_raw.replace("_", " ").lower().strip()

            intent_raw = intent_raw.replace("airquality", "air quality").strip()

            return [intent_raw]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        all_topics_dict = dict()
        all_intent2text = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = "\t"
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_topic_raw = str(raw_item[4]).lower().strip()
                if cur_topic_raw not in all_topics_dict:
                    all_topics_dict[cur_topic_raw] = 1
                else:
                    all_topics_dict[cur_topic_raw] += 1

                cur_intent_label = str(raw_item[1]).strip()
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                if len(_cur_intent_labels) == 0:
                    continue
                for _cur_intent_label in _cur_intent_labels:
                    if _cur_intent_label not in all_intents_dict:
                        all_intents_dict[_cur_intent_label] = 1
                    else:
                        all_intents_dict[_cur_intent_label] += 1

                    cur_text_raw = str(raw_item[3]).strip()
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
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = "\t"
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_topic_raw = str(raw_item[4]).lower().strip()
                    cur_text_raw = str(raw_item[3]).strip()
                    cur_intent_label = str(raw_item[1]).strip()

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
                        if cur_topic_raw == "recipes":
                            cur_topic_raw = "recipe"

                        cur_domain, cur_topic = "daily life", cur_topic_raw
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
