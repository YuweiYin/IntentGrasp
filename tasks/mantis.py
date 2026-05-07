# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMantis(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "mantis"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2019,  # The year of publication/preprint
            "paper_url": "https://arxiv.org/abs/1912.04639",  # URL of the dataset paper
            "license": None,  # the releasing license of the original dataset
            "intent_description": {
                "Further Details": "A user (either asking or answering user) provides more details.",
                "Follow Up Question": "Asking user asks one or more follow up questions about relevant issues.",
                "Information Request": "A user (either asking or answering user) is asking for clarifications or "
                                       "further information.",
                "Potential Answer": "A potential solution, provided by the answering user.",
                "Positive Feedback": "Asking user provides positive feedback about the offered solution.",
                "Negative Feedback": "Asking user provides negative feedback about the offered solution.",
                "Greetings / Gratitude": "A user (asking or answering user) offers a greeting or expresses gratitude.",
            },  # The description of each intent label
        }
        # Section 5: MANtIS
        #   Two expert annotators labelled each utterance within our sampled conversations;
        #   151 utterances were labelled by both annotators to determine the agreement between
        #     the annotators, leading to a Krippendorff's \alpha of 0.71.

        self.task_data = {
            "train": {
                "filenames": ["train.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "Follow Up Question": "To ask a follow-up question.",  # 435,
            "Further Details": "To provide further details.",  # 2311,
            "Greetings / Gratitude": "To express greetings or gratitude.",  # 845,
            "Information Request": "To request information.",  # 866,
            "Negative Feedback": "To give negative feedback.",  # 336,
            "Positive Feedback": "To give positive feedback.",  # 395,
            "Potential Answer": "To provide a potential answer.",  # 1553,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "Follow Up Question": ["smart assistant", "information seeking"],  # 435,
            "Further Details": ["smart assistant", "information seeking"],  # 2311,
            "Greetings / Gratitude": ["smart assistant", "information seeking"],  # 845,
            "Information Request": ["smart assistant", "information seeking"],  # 866,
            "Negative Feedback": ["smart assistant", "information seeking"],  # 336,
            "Positive Feedback": ["smart assistant", "information seeking"],  # 395,
            "Potential Answer": ["smart assistant", "information seeking"],  # 1553,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        intent_label2text = {
            # "OQ": "Opening Question",
            "FD": "Further Details",
            "FQ": "Follow Up Question",
            "IR": "Information Request",
            "PA": "Potential Answer",
            "PF": "Positive Feedback",
            "NF": "Negative Feedback",
            "GG": "Greetings / Gratitude",
            # "O": "Other",
        }

        def _normalize_intent_labels(intents_raw: list) -> List[str]:
            res_intents = []
            for intent_raw in intents_raw:
                assert isinstance(intent_raw, str)
                if intent_raw == "OQ":  # ignore the "Opening Question"
                    continue
                if intent_raw == "O":  # ignore "Other" intent
                    continue
                assert intent_raw in intent_label2text
                res_intents.append(intent_label2text[intent_raw])
            return res_intents

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
                assert file_format == "json"
                cur_data = DataIO.load_json(str(cur_filepath))
                cur_split_raw_data += [_item for _item in list(cur_data.values()) if "has_intent_labels" in _item]

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                for conv in raw_item["utterances"]:
                    if "intent" not in conv:  # Note: there is one bad data point
                        continue
                    _cur_intent_labels = conv["intent"]
                    for _cur_intent_label in _normalize_intent_labels(_cur_intent_labels):
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

        user_cnt, agent_cnt = 0, 0  # 2064, 3047

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
                cur_data = DataIO.load_json(str(cur_filepath))
                cur_file_raw_data = [_item for _item in list(cur_data.values()) if "has_intent_labels" in _item]

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_title_raw = str(raw_item["title"]).strip()  # title from the forum
                    cur_category_raw = str(raw_item["category"]).strip()  # domain to which the dialogue belongs

                    dialogue_history = f"Title: {cur_title_raw}".replace("\n", " ").strip()
                    for conv in raw_item["utterances"]:
                        cur_speaker = str(conv["actor_type"]).strip()  # conv["user_name"]
                        cur_utterance_raw = str(conv["utterance"]).strip()
                        cur_text_raw = f"{cur_speaker}: {cur_utterance_raw}".replace("\n", " ").strip()

                        cur_context_raw = dialogue_history
                        dialogue_history = f"{dialogue_history}\n{cur_text_raw}".strip()

                        if "intent" not in conv:  # Note: there is one bad data point
                            continue
                        cur_intent_labels = conv["intent"]
                        _cur_intent_labels = _normalize_intent_labels(cur_intent_labels)
                        if len(_cur_intent_labels) == 0:
                            continue

                        if cur_speaker == "user":
                            user_cnt += 1
                        elif cur_speaker == "agent":
                            agent_cnt += 1
                        else:
                            raise ValueError(f"Unknown speaker: {cur_speaker}")

                        for _cur_intent_label in _cur_intent_labels:
                            assert _cur_intent_label in cur_split_intents
                            cur_split_intents[_cur_intent_label] += 1

                        cur_category_raw = str(cur_category_raw).strip()
                        cur_category_raw = "ask ubuntu" if cur_category_raw == "askubuntu" else cur_category_raw
                        cur_category_raw = "world building" if cur_category_raw == "worldbuilding" else cur_category_raw

                        assert len(cur_context_raw) > 0
                        iu_context = f"### Chat History:\n{cur_context_raw}\n\n### Reply:\n{cur_text_raw}"
                        iu_question_raw = f"What is the intent of the {cur_speaker}'s reply?"
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
                            "paper_year": 2019,  # The year of publication/preprint
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
