# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskEmpatheticIntents(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "empathetic_intents"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2020.coling-main.429/",  # URL of the dataset paper
            # "To fill this gap, we have developed a taxonomy of empathetic listener intents by
            #   manually annotating around 500 utterances of the EmpatheticDialogues dataset."
            "license": "CC-BY",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Intents: (32-39)
        # agreeing, acknowledging, encouraging, consoling, sympathizing, suggesting, questioning, wishing
        # Emotions: (0-31 + 40)
        # afraid, angry, annoyed, anticipating, anxious, apprehensive, ashamed, caring, confident, content,
        # devastated, disappointed, disgusted, embarrassed, excited, faithful, furious, grateful, guilty, hopeful,
        # impressed, jealous, joyful, lonely, nostalgic, prepared, proud, sad, sentimental, surprised, terrified,
        # trusting, neutral

        # Section 4: Taxonomy of Empathetic Response Intents
        #   In this process, 20 dialogues belonging to each emotion were randomly selected and
        #     each sentence in all listener utterances were manually annotated by an expert evaluator with a label
        #     that best describes their intent. This resulted in 521 sentences manually annotated with intent labels.

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
            "acknowledging": "To express acknowledgment.",  # 774,
            "agreeing": "To agree.",  # 774,
            "consoling": "To express consolation.",  # 774,
            "encouraging": "To express encouragement.",  # 774,
            "questioning": "To ask a question.",  # 774,
            "suggesting": "To give suggestions.",  # 774,
            "sympathizing": "To express sympathy.",  # 774,
            "wishing": "To express a wish.",  # 774,
        }  # intent labels --> intent statements

        self.intent_label2category = {  # empathetic response intents
            "acknowledging": ["empathetic response", "acknowledging"],  # 774,
            "agreeing": ["empathetic response", "agreeing"],  # 774,
            "consoling": ["empathetic response", "consoling"],  # 774,
            "encouraging": ["empathetic response", "encouraging"],  # 774,
            "questioning": ["empathetic response", "questioning"],  # 774,
            "suggesting": ["empathetic response", "suggesting"],  # 774,
            "sympathizing": ["empathetic response", "sympathizing"],  # 774,
            "wishing": ["empathetic response", "wishing"],  # 774,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        intent_id2name = dict()
        intent_name2id = dict()
        intent_label_fp = os.path.join(self.raw_data_dir, self.task_name, "labels.txt")
        intent_label_list = DataIO.load_txt(str(intent_label_fp))
        for il in intent_label_list:
            assert isinstance(il, str) and "," in il
            il = il.strip()
            assert len(il.split(",")) == 2
            cur_label_name, cur_label_id = il.split(",")
            cur_label_name = str(cur_label_name).strip()
            cur_label_id = int(cur_label_id)

            intent_id2name[cur_label_id] = cur_label_name
            intent_name2id[cur_label_name] = cur_label_id

        def _normalize_intent_label(intent_id_raw: int) -> List[str]:
            assert intent_id_raw in intent_id2name
            intent_name_raw = intent_id2name[intent_id_raw]

            assert 32 <= intent_id_raw <= 39
            intent_clear = intent_name_raw

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
                assert file_format == "txt"
                cur_data = DataIO.load_txt(str(cur_filepath))
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                assert isinstance(raw_item, str) and "<SEP>" in raw_item and len(raw_item.split("<SEP>")) == 2
                cur_intent_label_id, cur_text_raw = raw_item.split("<SEP>")
                cur_intent_label_id = int(str(cur_intent_label_id).strip())

                assert 0 <= cur_intent_label_id <= 40
                if not (32 <= cur_intent_label_id <= 39):
                    continue  # only keep intents
                _cur_intent_labels = _normalize_intent_label(cur_intent_label_id)
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
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    assert isinstance(raw_item, str) and "<SEP>" in raw_item and len(raw_item.split("<SEP>")) == 2
                    cur_intent_label_id, cur_text_raw = raw_item.split("<SEP>")
                    cur_intent_label_id = int(str(cur_intent_label_id).strip())
                    cur_text_raw = str(cur_text_raw).strip()

                    assert 0 <= cur_intent_label_id <= 40
                    if not (32 <= cur_intent_label_id <= 39):
                        continue  # only keep intents
                    _cur_intent_labels = _normalize_intent_label(cur_intent_label_id)
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "speaker"
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
