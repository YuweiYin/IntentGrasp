# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMathDial(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "mathdial"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": True,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2025,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2025.bea-1.63/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset (ACL publication - CC-BY)
            "intent_description": {},  # The description of each intent label
        }
        # Section 3: Re-annotating the MathDial Dataset -- rule-based procedure + GPT-4o-based annotation pipeline

        self.task_data = {
            "train": {
                # "filenames": ["train.tsv"],
                "filenames": [],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                # "filenames": ["valid.tsv"],
                "filenames": [],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["train.tsv", "valid.tsv", "test.tsv"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            # "revealing strategy": "To reveal a proper problem-solving strategy to the student.",  # 1141,
            "revealing answer": "To reveal the correct answer to the student or directly point out the mistakes.",  # 895,
            # "guiding student focus": "To guide the focus of the student towards the correct direction.",  # 687,
            "guiding student focus": "To provide the student with the correct direction, "
                                     "to guide the student to self-correction, or "
                                     "to give a hint about a proper problem-solving strategy.",  # 687,
            # "seek strategy": "To guide the student to seek a proper problem-solving strategy.",  # 658,
            "asking for explanation": "To ask the student to explain their own solution or thoughts.",  # 653,
            # "seeking self correction": "To ask the student to seek self-correction.",  # 643,
            # "seeking world knowledge": "To ask the student to recall relevant world knowledge.",  # 257,
            # "greeting / farewell": "To express a greeting or a farewell.",  # 217,
            # "recall relevant information": "To ask the student to revisit the question or recall relevant information.",  # 93,
            "perturbing the question": "To perturb the original question and test the student's understanding.",  # 89,
            # "general inquiry": "",  # 40,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            # "revealing strategy": ["teaching", "math"],  # 1141,
            "revealing answer": ["teaching", "math"],  # 895,
            "guiding student focus": ["teaching", "math"],  # 687,
            # "seek strategy": ["teaching", "math"],  # 658,
            "asking for explanation": ["teaching", "math"],  # 653,
            # "seeking self correction": ["teaching", "math"],  # 643,
            # "seeking world knowledge": ["teaching", "math"],  # 257,
            # "greeting / farewell": ["teaching", "math"],  # 217,
            # "recall relevant information": ["teaching", "math"],  # 93,
            "perturbing the question": ["teaching", "math"],  # 89,
            # "general inquiry": ["teaching", "math"],  # 40,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "general inquiry", "greeting / farewell", "revealing strategy",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_clear = intent_raw.strip()
            intent_clear = intent_clear.replace("_", " ").strip()
            intent_clear = intent_clear.replace("-", " ").strip()
            intent_clear = intent_clear.replace("/", " ").strip()
            intent_clear = intent_clear.replace("  ", " ").strip()
            intent_clear = intent_clear.replace("  ", " ").strip()
            intent_clear = intent_clear.lower().strip()

            intent_clear = intent_clear.replace("fairwell", "farewell").strip()  # A type in the raw data
            if intent_clear == "greeting farewell":
                intent_clear = "greeting / farewell"

            if intent_clear == "seeking world knowledge" or intent_clear == "recall relevant information":
                intent_clear = "guiding student focus"

            if intent_clear == "seek strategy" or intent_clear == "seeking self correction":
                intent_clear = "guiding student focus"

            return [intent_clear]

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
                assert file_format == "tsv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = "\t"
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                assert isinstance(raw_item, list) and len(raw_item) == 16
                cur_intent_label = str(raw_item[-1]).strip()
                if len(cur_intent_label) == 0:
                    continue

                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                if len(_cur_intent_labels) == 0:
                    continue
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
                assert file_format == "tsv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = "\t"
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    assert isinstance(raw_item, list) and len(raw_item) == 16
                    cur_intent_label = str(raw_item[-1]).strip()
                    if len(cur_intent_label) == 0:
                        continue

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker_raw = str(raw_item[2]).lower().strip()  # either the teacher or student
                    assert cur_speaker_raw == "teacher"
                    cur_text_raw = str(raw_item[4]).replace("\n", " ").strip()  # the speech text
                    cur_math_question = str(raw_item[7]).replace("\n", " ").strip()  # math question
                    cur_correct_sol = str(raw_item[8]).replace("\n", " ").strip()  # the correct solution to the question
                    cur_student_sol = str(raw_item[9]).replace("\n", " ").strip()  # erroneous solution by the student

                    cur_context_raw = (f"### Math Question: {cur_math_question}\n\n"
                                       f"### Correct Solution:\n{cur_correct_sol}\n\n"
                                       f"### Student's Incorrect Solution:\n{cur_student_sol}")

                    cur_speaker = cur_speaker_raw
                    iu_context = f"{cur_context_raw}\n\n### Teacher's Response: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the teacher's response to the student?"
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
                        "paper_year": 2025,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "dialogue",  # str: query/dialogue/monologue
                        "intent_type": "single",  # "multiple" if multiple intents per item else "single"
                        "is_synthetic": True,  # True if the dataset is synthetic (not human annotated)
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
