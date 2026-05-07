# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMoralStories(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "moral_stories"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2021,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2021.emnlp-main.54/",  # URL of the dataset paper
            "license": "MIT",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 4 - Table 5: Human Evaluation - Intention = 0.94
        # Appendix D: Generation: Supplementary Details
        #   Inter-rater agreement scores for each task and setting, based on the binarized ratings,
        #     are given in Tables 17 - 19 as percentage agreement, i.e. the fraction of stories for
        #     which all three raters gave the same rating. (Table 17: Intention = 58.0%)
        #   Agreement scores computed according to Krippendorff's \alpha were found to be unreliable
        #     due to the sparsity of annotations (most samples were evaluated by a different set of annotators,
        #     due to the nature of crowdsourcing) and the skewness of the collected ratings (most scores
        #     fall inside the 3-5 range, especially for coherence). For clarity and due to space limitations,
        #     we do not include the corresponding scores, but are happy to provide them on request.

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

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            assert "wants to" in intent_raw or "wanted to" in intent_raw or "needs to" in intent_raw
            if "wants to" in intent_raw:
                intent_clear = intent_raw.split("wants to")[1].strip()
                intent_clear = f"To {intent_clear}"
            elif "wanted to" in intent_raw:
                intent_clear = intent_raw.split("wanted to")[1].strip()
                intent_clear = f"To {intent_clear}"
            elif "needs to" in intent_raw:
                intent_clear = intent_raw.split("needs to")[1].strip()
                intent_clear = f"To {intent_clear}"
            else:
                raise ValueError(f"Invalid intent label: {intent_raw}")

            return [intent_clear.strip()]

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
                assert file_format == "jsonl"
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_moral_label = int(raw_item["label"])
                assert cur_moral_label == 0 or cur_moral_label == 1
                if cur_moral_label == 0:
                    continue  # only consider moral case (label=1), where the action faithfully reflects the intention
                assert "moral_action" in raw_item and "moral_consequence" in raw_item

                cur_situation = str(raw_item["situation"]).strip()
                cur_character_name = cur_situation.split()[0].strip()
                if cur_character_name.endswith("'s"):
                    cur_character_name = cur_character_name.rstrip("'s")
                assert len(cur_character_name) > 0
                cur_moral_action = str(raw_item["moral_action"]).strip()
                if not cur_moral_action.startswith(cur_character_name):
                    continue

                cur_intent_label = str(raw_item["intention"]).strip()
                if not ("wants to" in cur_intent_label or
                        "wanted to" in cur_intent_label or
                        "needs to" in cur_intent_label):
                    continue
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
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

        self.intent_label2statement = {_i: _i for _i in all_intents_list}

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
                    cur_moral_label = int(raw_item["label"])
                    assert cur_moral_label == 0 or cur_moral_label == 1
                    if cur_moral_label == 0:
                        continue  # only consider moral case, where the action faithfully reflects the intention
                    assert "moral_action" in raw_item and "moral_consequence" in raw_item

                    cur_situation = str(raw_item["situation"]).replace("\n", " ").strip()
                    cur_moral_action = str(raw_item["moral_action"]).replace("\n", " ").strip()
                    cur_character_name = cur_situation.split()[0].strip()
                    if cur_character_name.endswith("'s"):
                        cur_character_name = cur_character_name.rstrip("'s")
                    assert len(cur_character_name) > 0
                    if not cur_moral_action.startswith(cur_character_name):
                        continue

                    cur_intent_label = str(raw_item["intention"]).strip()
                    if not ("wants to" in cur_intent_label or
                            "wanted to" in cur_intent_label or
                            "needs to" in cur_intent_label):
                        continue
                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    assert len(_cur_intent_labels) == 1
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = cur_character_name
                    iu_context = f"### Situation: {cur_situation}\n### Action: {cur_moral_action}"
                    iu_question_raw = f"What is the intent of {cur_speaker}?"
                    iu_answer_intent_raw = _cur_intent_labels

                    cur_domain_topic = []
                    cur_domain, cur_topic = "daily life", ""  # "everyday events", ""
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
