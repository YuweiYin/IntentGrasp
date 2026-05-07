# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskConda(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "conda"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2021.findings-acl.213/",  # URL of the dataset paper
            # "The annotators manually classified the utterances into four labels:
            #   E (Explicit toxicity), I (Implicit toxicity), A (Action) and O (Other)."
            "license": "CC-BY",  # the releasing license of the original dataset (ACL publication - CC-BY)
            "intent_description": {
                "E": "expressing explicit toxicity",  # "explicit"
                "I": "expressing implicit toxicity",  # "implicit"
                "A": "taking toxic or negative action",  # "action"
                "O": "no clear toxicity targeting others",  # "other"
            },  # The description of each intent label
            # Explicit toxicity: Typically contains toxic word(s). The intent is to insult or humiliate others, or to
            #     make others want to leave the conversation or quit the game. There is no need to consider the context.
            # Implicit toxicity: Hidden toxicity that normally cannot be seen from the text itself. The text might be
            #     factual or even positive (e.g. sarcasm). However, based on the utterance or conversation context,
            #         the intent of insulting or humiliating others can be inferred. Typically, contains no toxic word.
            # Action: Doesn't belong to I or E, but contains an action such as report, commend, pause, stop, or exit game.
            # Other: Doesn't belong to I or E or A. May or may not contain toxic words. Includes curses,
            #     self-deprecation or any other emotional expression that is NOT targeted at others.
        }
        # Section 3.3: Annotation
        #   Overall, we observed that the agreement measure for utterance classification was higher for
        #     gamer annotators only (Fleiss' kappa = 0.785) versus the whole group (Fleiss' kappa = 0.755).
        #   The lower inter-rater agreement in the whole group is because non-gamer annotators have
        #     low understanding of the game context and domain-specific language.

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
            "expressing explicit toxicity": "To express explicit toxicity.",  # 4711,
            "expressing implicit toxicity": "To express implicit toxicity.",  # 2274,
            "no clear toxicity targeting others": "No clear toxicity targeting others.",  # 26611,
            "taking toxic or negative action": "To take toxic or negative action.",  # 2299,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "expressing explicit toxicity": ["toxic speech", "abusive language"],  # 4711,
            "expressing implicit toxicity": ["toxic speech", "abusive language"],  # 2274,
            "no clear toxicity targeting others": ["toxic speech", "abusive language"],  # 26611,
            "taking toxic or negative action": ["toxic speech", "abusive language"],  # 2299,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        index_label2name = {
            "E": "expressing explicit toxicity",  # "explicit"
            "I": "expressing implicit toxicity",  # "implicit"
            "A": "taking toxic or negative action",  # "action"
            "O": "no clear toxicity targeting others",  # "other"
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            return [intent_raw.strip()]

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
                assert file_format == "csv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = ","
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                # ["Id", "matchId", "conversationId", "utterance", "chatTime",
                # "playerSlot", "playerId", "intentClass", "slotClasses", "slotTokens"]
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item[7]).strip()
                assert cur_intent_label in index_label2name
                cur_intent_label = index_label2name[cur_intent_label]
                for _cur_intent_label in _normalize_intent_label(cur_intent_label):
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
                    cur_text_raw = str(raw_item[3]).strip()
                    cur_intent_label = str(raw_item[7]).strip()

                    assert cur_intent_label in index_label2name
                    cur_intent_label = index_label2name[cur_intent_label]

                    assert cur_intent_label in cur_split_intents
                    cur_split_intents[cur_intent_label] += 1

                    cur_speaker = "game player"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = (f"Considering the toxicity in speech or action, "
                                       f"what is the intent of the {cur_speaker}?")
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
                        "paper_year": 2021,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "monologue",  # str: query/dialogue/monologue
                        "intent_type": "single",  # "multiple" if multiple intents per item else "single"
                        "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
                        "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
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
