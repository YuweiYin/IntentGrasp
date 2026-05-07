# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskI2Hate(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "i2hate"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2026,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2026.eacl-short.8/",  # URL of the dataset paper
            "license": "CC-BY-SA",  # the releasing license of the original dataset (ACL publication - CC-BY-NC-SA)
            "intent_description": {
                # The Intent dimension captures seven distinct speaker motivations behind hate speech production:
                "Affective Aggression": "Reactive emotional expression driven by anger, frustration, or outrage, "
                                        "characterized by impulsive hostile language without strategic planning.",
                # Affective Aggression -- As Mane et al. (2025) note, “the Frustration-Aggression Theory suggests that
                # frustration leads to aggression, which can be exacerbated by social media platforms.”
                "Derisive Trolling": "Deliberate provocation for amusement or disruption, employing mockery, sarcasm, "
                                     "or feigned ignorance to elicit reactions.",
                "Dominance & Subjugation": "Assertion of power and social hierarchy through degradation and "
                                           "belittling language that positions target groups as inferior.",
                "Ideological Expression": "Articulation of hateful worldviews or political ideologies that "
                                          "position certain groups as threats to valued institutions or social order.",
                "Performative Reinforcement": "In-group signaling and solidarity building through "
                                              "shared hateful rhetoric, reinforcing group identity and boundaries.",
                "Strategic Incitement": "Calculated language crafted to achieve specific political, ideological, or "
                                        "social objectives, including mobilizing followers or "
                                        "coordinating hostile actions.",
                "Threat & Intimidation": "Direct or implied threats designed to instill fear, silence targets, or "
                                         "warn of impending harm.",
            },  # The description of each intent label
        }
        # Section 2.1: Data Collection and Annotation
        #   Three annotators underwent rigorous two-week training on the taxonomy framework and completed practice
        #     annotation rounds before independently labeling all posts (detailed methodology in Appendix A).
        #   Inter-annotator agreement measured by Fleiss' kappa yielded \kappa=0.74 for Intent
        #     and \kappa=0.79 for Impact, indicating substantial agreement.

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
                "filenames": ["test.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "Affective Aggression": "To express affective aggression, which is a reactive emotional expression "
                                    "driven by anger, frustration, or outrage, characterized by impulsive, hostile "
                                    "language without strategic planning.",  # 442,
            "Derisive Trolling": "To deliberately provoke for amusement or disruption, employing mockery, sarcasm, "
                                 "or feigned ignorance to elicit reactions.",  # 383,
            "Dominance & Subjugation": "To assert power and social hierarchy through degradation and belittling "
                                       "language that positions target groups as inferior.",  # 520,
            "Ideological Expression": "To articulate hateful worldviews or political ideologies that "
                                      "position certain groups as threats to valued institutions or social order.",  # 719,
            "Performative Reinforcement": "To build solidarity through shared hateful rhetoric, "
                                          "reinforcing group identity and boundaries.",  # 424,
            "Strategic Incitement": "To achieve specific political, ideological, or social objectives, "
                                    "including mobilizing followers or coordinating hostile actions.",  # 754,
            "Threat & Intimidation": "To threaten to instill fear, silence targets, or warn of impending harm.",  # 445,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: List[str]) -> List[str]:
            res_intents = [_intent.strip() for _intent in intent_raw if len(_intent.strip()) > 0]
            return res_intents

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        skip_cnt = 0  # skip_cnt == 16
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "parquet"
                cur_data_df = DataIO.load_parquet(str(cur_filepath))
                # "Text", "Intent Labels", "Impact Labels", "Sample ID"
                cur_data_list = []
                for _t, _i in zip(cur_data_df["Text"].tolist(), cur_data_df["Intent Labels"].tolist()):
                    _t, _i = str(_t).strip(), str(_i).strip()
                    if len(_t) == 0 or len(_i) == 0:
                        skip_cnt += 1
                        continue
                    cur_data_list.append({
                        "text": str(_t).strip(),
                        "intent": str(_i).strip(),
                    })
                cur_split_raw_data += cur_data_list

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label_raw = str(raw_item["intent"]).strip()
                if "," in cur_intent_label_raw:
                    _cur_intent_labels = cur_intent_label_raw.split(",")
                else:
                    _cur_intent_labels = [cur_intent_label_raw]
                _cur_intent_labels = _normalize_intent_label(_cur_intent_labels)
                if len(_cur_intent_labels) == 0:
                    continue

                for _cur_intent_label in _cur_intent_labels:
                    assert _cur_intent_label in self.intent_label2statement
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
                assert file_format == "parquet"
                cur_data_df = DataIO.load_parquet(str(cur_filepath))
                # "Text", "Intent Labels", "Impact Labels", "Sample ID"
                cur_data_list = []
                for _t, _i in zip(cur_data_df["Text"].tolist(), cur_data_df["Intent Labels"].tolist()):
                    _t, _i = str(_t).strip(), str(_i).strip()
                    if len(_t) == 0 or len(_i) == 0:
                        skip_cnt += 1
                        continue
                    cur_data_list.append({
                        "text": str(_t).strip(),
                        "intent": str(_i).strip(),
                    })
                cur_file_raw_data = cur_data_list

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item["text"]).strip()
                    cur_intent_label_raw = str(raw_item["intent"]).strip()
                    if "," in cur_intent_label_raw:
                        _cur_intent_labels = cur_intent_label_raw.split(",")
                    else:
                        _cur_intent_labels = [cur_intent_label_raw]
                    _cur_intent_labels = _normalize_intent_label(_cur_intent_labels)
                    if len(_cur_intent_labels) == 0:
                        continue

                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "poster"  # Twitter poster
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = (f"By posting the above content on social media, "
                                       f"what is the intent of the {cur_speaker}?")
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []
                    cur_domain, cur_topic = "toxic speech", "hate speech"
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
                        "paper_year": 2026,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "monologue",  # str: query/dialogue/monologue
                        "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
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
