# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskXSID(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "xsid"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2021.naacl-main.197/",  # URL of the dataset paper
            "license": "CC-BY-SA",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 2.2: xSID
        #   We choose to use the Snips and Facebook (Schuster et al., 2019) data as a starting point.
        #   Ultimately, the data collection process proceeded in two steps: translation of the data from English,
        #     and slot annotation in the target language.
        #   We calculated inter-annotator agreement for the guidelines; three annotators native in Dutch annotated
        #     100 samples, and reached a Fleiss Kappa (Fleiss, 1971) score of 0.924, which is very high agreement.

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
            "weather: find": "To get the weather information.",  # 13067,
            "alarm: set alarm": "To set the alarm.",  # 4278,
            "reminder: set reminder": "To set the reminder.",  # 4206,
            "play music": "To play music.",  # 1949,
            "book restaurant": "To book a restaurant.",  # 1937,
            "search creative work": "To search for a creative work.",  # 1903,
            "add to playlist": "To add music to the playlist.",  # 1889,
            "rate book": "To rate a book.",  # 1873,
            "search screening event": "To search for screening events.",  # 1824,
            "alarm: cancel alarm": "To cancel the alarm.",  # 1480,
            "reminder: cancel reminder": "To cancel the reminder.",  # 877,
            "alarm: show alarms": "To show the alarms.",  # 806,
            "reminder: show reminders": "To show the reminders.",  # 680,
            "alarm: modify alarm": "To midify the alarm.",  # 407,
            "alarm: time left on alarm": "To check the time left on the alarm.",  # 302,
            "alarm: snooze alarm": "To snooze the alarm.",  # 300,
            "weather: check sunset": "To check the sunset information.",  # 119,
            "weather: check sunrise": "To check the sunrise information.",  # 88,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "weather: find": ["daily life", "weather"],  # 13067,
            "alarm: set alarm": ["daily life", "alarm"],  # 4278,
            "reminder: set reminder": ["daily life", "reminder"],  # 4206,
            "play music": ["daily life", "music"],  # 1949,
            "book restaurant": ["daily life", "restaurant"],  # 1937,
            "search creative work": ["daily life", "searching"],  # 1903,
            "add to playlist": ["daily life", "music"],  # 1889,
            "rate book": ["daily life", "book"],  # 1873,
            "search screening event": ["daily life", "searching"],  # 1824,
            "alarm: cancel alarm": ["daily life", "alarm"],  # 1480,
            "reminder: cancel reminder": ["daily life", "reminder"],  # 877,
            "alarm: show alarms": ["daily life", "alarm"],  # 806,
            "reminder: show reminders": ["daily life", "reminder"],  # 680,
            "alarm: modify alarm": ["daily life", "alarm"],  # 407,
            "alarm: time left on alarm": ["daily life", "alarm"],  # 302,
            "alarm: snooze alarm": ["daily life", "alarm"],  # 300,
            "weather: check sunset": ["daily life", "weather"],  # 119,
            "weather: check sunrise": ["daily life", "weather"],  # 88,
        }  # intent labels --> intent categories (domains & topics)

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
            text_set = set()
            _cur_text, _cur_intent = "", ""
            for txt_line in txt_raw:
                txt_line = str(txt_line).strip()
                if len(txt_line) == 0:
                    # Done with one instance
                    if _cur_text not in text_set:
                        text_set.add(_cur_text)
                        txt_raw_instances.append([_cur_text, _cur_intent])
                    _cur_text, _cur_intent = "", ""  # reset the text and intents
                elif txt_line.startswith("# text:"):
                    _cur_text = txt_line[len("# text:"):].strip()
                elif txt_line.startswith("# intent:"):
                    _cur_intent = txt_line[len("# intent:"):].strip()
                else:
                    pass
            return txt_raw_instances

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            intent_clear = ""
            for ch in intent_raw:
                if ch == "/":
                    intent_clear += ": "
                elif ch == "_":
                    intent_clear += " "
                elif ch.isupper():
                    intent_clear += f" {ch.lower()}"
                else:
                    intent_clear += ch

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
                assert file_format == "txt"
                cur_data = DataIO.load_txt(str(cur_filepath))
                cur_data_instances = _process_raw_txt(cur_data)
                cur_split_raw_data += cur_data_instances

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                assert isinstance(raw_item, list) and len(raw_item) == 2
                cur_intent_label = str(raw_item[1]).strip()
                cur_intent_label = _normalize_intent_label(cur_intent_label)[0].strip()
                if ":" in cur_intent_label:
                    cur_topic_raw = cur_intent_label.split(":")[0].strip()
                else:
                    cur_topic_raw = cur_intent_label

                if cur_intent_label not in all_intents_dict:
                    all_intents_dict[cur_intent_label] = 1
                else:
                    all_intents_dict[cur_intent_label] += 1

                intent2domain[cur_intent_label] = cur_topic_raw
                if cur_topic_raw not in domain2intent:
                    domain2intent[cur_topic_raw] = [cur_intent_label]
                else:
                    domain2intent[cur_topic_raw].append(cur_intent_label)

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
                    cur_intent_label = _normalize_intent_label(cur_intent_label)[0].strip()

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
                    # cur_domain_topic.append(self.intent_label2category[cur_intent_label])
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
