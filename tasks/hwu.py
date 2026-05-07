# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskHWU(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "hwu"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://link.springer.com/chapter/10.1007/978-981-15-9323-9_15",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label (Note: similar to the SLURP dataset)
        }
        # Section 4.2: Annotation & Inter-annotator Agreement
        #   Since there was a predetermined set of Intents for which we collected data,
        #     there was no need for separate Intent annotations (some Intent corrections were needed).

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
                "filenames": ["data_full.csv"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "calendar: set": "To set the calendar.",  # 1451,
            "play: music": "To play music.",  # 1205,
            # "general: quirky": "",  # 1088,
            "weather: query": "To ask about the weather.",  # 1062,
            "qa: factoid": "To ask about factoid.",  # 1052,
            "calendar: query": "To check the calendar.",  # 1002,
            # "general: negate": "",  # 939,
            "news: query": "To check the news.",  # 877,
            # "general: praise": "",  # 785,
            "email: query": "To check the email.",  # 759,
            "email: sendemail": "To send an email.",  # 694,
            # "general: explain": "",  # 684,
            "datetime: query": "To check the datetime.",  # 626,
            # "general: repeat": "",  # 585,
            # "general: affirm": "",  # 554,
            "play: radio": "To play radio.",  # 551,
            # "general: confirm": "",  # 550,
            "social: post": "To post social events.",  # 541,
            "calendar: remove": "To remove events from the calendar.",  # 533,
            "qa: definition": "To ask about the definition.",  # 504,
            # "general: dont care": "",  # 450,
            "cooking: recipe": "To ask or talk about the cooking recipe.",  # 415,
            "transport: query": "To check the transport.",  # 399,
            "play: podcasts": "To play podcasts.",  # 379,
            "qa: currency": "To ask about the currency.",  # 378,
            "lists: query": "To check the lists.",  # 370,
            "lists: remove": "To remove items from the lists.",  # 330,
            "recommendation: events": "To ask about event recommendations.",  # 324,
            # "general: command stop": "",  # 320,
            "alarm: set": "To set the alarm.",  # 297,
            "lists: create or add": "To create or add items to the lists.",  # 294,
            "music: query": "To search for the music.",  # 276,
            "qa: stock": "To ask about the stock.",  # 270,
            "recommendation: locations": "To ask about location recommendation.",  # 257,
            "iot: hue light off": "To turn off the light.",  # 246,
            "play: audiobook": "To play audiobook.",  # 241,
            "transport: ticket": "To book a ticket or ask about related information.",  # 239,
            "play: game": "To play game.",  # 237,
            "iot: hue light change": "To change the light color.",  # 224,
            "email: querycontact": "To check the email contact.",  # 221,
            "takeaway: query": "To check the takeaway information.",  # 215,
            "music: likeness": "To express likeness to the music.",  # 204,
            "alarm: query": "To check the alarm.",  # 203,
            "transport: traffic": "To ask about the traffic information.",  # 200,
            "takeaway: order": "To order takeaways.",  # 199,
            "iot: coffee": "To ask for some coffee.",  # 198,
            "social: query": "To check the news on social media.",  # 186,
            "transport: taxi": "To book a taxi.",  # 185,
            "iot: cleaning": "To ask to do the cleaning.",  # 172,
            "qa: maths": "To ask about math problems.",  # 166,
            "audio: volume mute": "To mute or turn off the audio.",  # 163,
            "audio: volume up": "To turn up the audio volume.",  # 145,
            "iot: hue light up": "To turn up the light.",  # 142,
            "iot: hue light dim": "To turn down the light.",  # 126,
            "alarm: remove": "To remove the alarm.",  # 123,
            # "general: joke": "",  # 122,
            "recommendation: movies": "To ask about movie recommendations.",  # 112,
            "iot: wemo off": "To turn off the smart plug or socket.",  # 100,
            "datetime: convert": "To convert the datetime.",  # 97,
            "email: addcontact": "To add an email contact.",  # 90,
            "music: settings": "To change the music settings.",  # 80,
            "audio: volume down": "To turn down the audio volume.",  # 80,
            "iot: wemo on": "To turn on the smart plug or socket.",  # 80,
            "iot: hue light on": "To turn on the light.",  # 39,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "general: quirky",  "general: negate", "general: praise", "general: explain", "general: repeat",
            "general: affirm", "general: confirm", "general: dont care", "general: command stop", "general: joke"
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            intent_clear = ""
            for ch in intent_raw:
                if ch.isupper():
                    intent_clear += f" {ch.lower()}"
                else:
                    intent_clear += ch

            intent_clear = intent_clear.replace("_", " ").strip()
            intent_clear = intent_clear.replace(" other", " ").strip()
            intent_clear = intent_clear.replace("commandstop", "command stop").strip()
            intent_clear = intent_clear.replace("dontcare", "dont care").strip()
            intent_clear = intent_clear.replace("lightchange", "light change").strip()
            intent_clear = intent_clear.replace("lightdim", "light dim").strip()
            intent_clear = intent_clear.replace("lightoff", "light off").strip()
            intent_clear = intent_clear.replace("lighton", "light on").strip()
            intent_clear = intent_clear.replace("lightup", "light up").strip()
            intent_clear = intent_clear.replace("createoradd", "create or add").strip()

            return [intent_clear.strip()]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        all_topics_dict = dict()
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
                    table_delimiter = ";"
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                # ["userid", "answerid", "scenario", "intent", "status",
                # "answer_annotation", "notes", "suggested_entities", "answer_normalised", "answer",
                # "question"]
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_topic_raw = str(raw_item[2]).strip()
                cur_intent_label = str(raw_item[3]).strip()

                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                assert len(_cur_intent_labels) == 1
                cur_intent_label = _cur_intent_labels[0]
                cur_intent_label = f"{cur_topic_raw}: {cur_intent_label}"
                if cur_intent_label in ["audio: volume", "cooking: query", "general: greet", "music: dislikeness"]:
                    continue  # skip 4 extra intents --> only keep 64 intents, as in https://arxiv.org/pdf/1903.05566
                if cur_intent_label in skip_intents:
                    continue

                if cur_intent_label not in all_intents_dict:
                    all_intents_dict[cur_intent_label] = 1
                else:
                    all_intents_dict[cur_intent_label] += 1

                if cur_topic_raw not in all_topics_dict:
                    all_topics_dict[cur_topic_raw] = 1
                else:
                    all_topics_dict[cur_topic_raw] += 1

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
                    table_delimiter = ";"
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item[9]).strip()
                    cur_topic_raw = str(raw_item[2]).strip()
                    cur_intent_label = str(raw_item[3]).strip()

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    assert len(_cur_intent_labels) == 1
                    cur_intent_label = _cur_intent_labels[0]
                    cur_intent_label = f"{cur_topic_raw}: {cur_intent_label}"
                    if cur_intent_label in ["audio: volume", "cooking: query", "general: greet", "music: dislikeness"]:
                        continue  # skip 4 extra intents --> keep 64 intents, as in https://arxiv.org/pdf/1903.05566
                    if cur_intent_label in skip_intents:
                        continue
                    assert cur_intent_label in cur_split_intents
                    cur_split_intents[cur_intent_label] += 1

                    cur_speaker = "user"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    assert cur_intent_label in self.intent_label2statement
                    iu_answer_intent_raw.append(self.intent_label2statement[cur_intent_label])

                    cur_domain_topic = []
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
