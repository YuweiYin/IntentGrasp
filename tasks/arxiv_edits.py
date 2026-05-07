# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskArxivEdits(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "arxiv_edits"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2022,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2022.emnlp-main.641/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset (ACL publication - CC-BY)
            "intent_description": {
                "improve content": "Update large amount of scientific content, add or delete major fact.",
                "improve grammar / fix typo": "Fix grammatical errors, correct typos, or smooth out grammar "
                                              "needed by other changes.",
                "improve format": "Adjust table, figure, equation, reference, citation, and punctuation etc.",
                "improve language: more accuracy / specific": "Minor adjustment to improve the accuracy or "
                                                              "specificness of the description.",
                "improve language: improve style": "Make the text sound more professional or coherent "
                                                   "without altering the meaning.",
                "improve language: readability / simplify": "Simplify complex concepts or delete redundant content "
                                                            "to improve readability.",
            },  # The description of each intent label
        }
        # Section 2: Constructing arXivEdits Corpus
        #   We manually annotate insertion, deletion, substitution, and derive reordering automatically,
        #     since it can be reliably found by heuristics.
        #   Therefore, instead of crowdsourcing, we hire two experienced in-house annotators to annotate
        #     the intention for 2,122 edits in 1,000 sentence revisions. A two-hour training session is provided to
        #     both annotators, during which they are asked to annotate 100 sentence pairs and discuss until consensus.
        #   The inter-annotator agreement is 0.67 measured by Cohen Kappa,
        #     and 0.81 if collapsing the Improve Language category.

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
            "improve content": "To update a large amount of scientific content, add or delete a major fact.",  # 620,
            "improve format": "To adjust table, figure, equation, reference, citation, and punctuation, etc.",  # 365,
            "improve grammar / fix typo": "To fix grammatical errors, correct typos, or "
                                          "smooth out grammar needed by other changes.",  # 545,
            "improve language: improve style": "To make the text sound more professional or coherent "
                                               "without altering the meaning.",  # 185,
            "improve language: more accuracy / specific": "To make a minor adjustment to improve the accuracy or "
                                                          "specificness of the description.",  # 252,
            "improve language: readability / simplify": "To simplify complex concepts or delete redundant content "
                                                        "to improve readability.",  # 166,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "improve content": ["writing", "edit"],  # 620,
            "improve format": ["writing", "edit"],  # 365,
            "improve grammar / fix typo": ["writing", "edit"],  # 545,
            "improve language: improve style": ["writing", "edit"],  # 185,
            "improve language: more accuracy / specific": ["writing", "edit"],  # 252,
            "improve language: readability / simplify": ["writing", "edit"],  # 166,
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            intent_clear = ""
            for ch in intent_raw:
                if ch.isupper():
                    intent_clear += f" {ch.lower()}"
                else:
                    intent_clear += ch

            intent_clear = intent_clear.replace("spefific", "specific")  # A typo in the original dataset
            intent_clear = intent_clear.replace("_", " ").strip()
            intent_clear = intent_clear.replace("-", " ").strip()
            intent_clear = intent_clear.replace("  ", " ").strip()

            if intent_clear == "content":
                intent_clear = "improve content"
            if intent_clear == "format":
                intent_clear = "improve format"
            if intent_clear == "improve grammar typo":
                intent_clear = "improve grammar / fix typo"
            if intent_clear == "lang accurate specific":
                intent_clear = "improve language: more accuracy / specific"
            if intent_clear == "lang professional improve style":
                intent_clear = "improve language: improve style"
            if intent_clear == "lang improve readability simplify":
                intent_clear = "improve language: readability / simplify"

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
                assert file_format == "json"
                cur_data = DataIO.load_json(str(cur_filepath))
                assert isinstance(cur_data, dict)
                cur_data = list(cur_data.values())
                cur_data = [_d for _d in cur_data if str(_d["sentence-1"]).lower().strip() !=
                            str(_d["sentence-2"]).lower().strip()]
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_sent1 = str(raw_item["sentence-1"])  # .strip()
                cur_sent2 = str(raw_item["sentence-2"])  # .strip()
                assert len(cur_sent1) > 0 and len(cur_sent2) > 0 and cur_sent1 != cur_sent2
                _cur_intent_labels = []
                for comb_name in ["edits-combination-0", "edits-combination-1", "edits-combination-2"]:
                    cur_edits_comb = dict(raw_item[comb_name])
                    cur_edits_intents = []
                    for edit_dict in cur_edits_comb.values():
                        assert isinstance(edit_dict, dict)
                        cur_intent_str = str(edit_dict["intention"]).strip()
                        cur_intent_str = _normalize_intent_label(cur_intent_str)[0].strip()
                        if cur_intent_str not in ["none", "lang other"]:  # skip unclear editing intents
                            cur_edits_intents.append(cur_intent_str)
                    _cur_intent_labels += cur_edits_intents
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

        edit_types = {"insertion", "deletion", "substitute"}

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
                assert isinstance(cur_data, dict)
                cur_data = list(cur_data.values())
                cur_data = [_d for _d in cur_data if str(_d["sentence-1"]).lower().strip() !=
                            str(_d["sentence-2"]).lower().strip()]
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_sent1 = str(raw_item["sentence-1"])  # .strip()
                    cur_sent2 = str(raw_item["sentence-2"])  # .strip()
                    assert len(cur_sent1) > 0 and len(cur_sent2) > 0 and cur_sent1 != cur_sent2

                    cur_sent1_tokens = cur_sent1.strip().split()
                    cur_sent2_tokens = cur_sent2.strip().split()

                    for comb_name in ["edits-combination-0", "edits-combination-1", "edits-combination-2"]:
                        cur_edits_comb = dict(raw_item[comb_name])
                        for edit_dict in cur_edits_comb.values():
                            assert isinstance(edit_dict, dict)
                            cur_intent_str = str(edit_dict["intention"]).strip()
                            cur_intent_str = _normalize_intent_label(cur_intent_str)[0].strip()
                            if cur_intent_str in ["none", "lang other"]:  # skip unclear editing intents
                                continue
                            assert cur_intent_str in cur_split_intents
                            cur_split_intents[cur_intent_str] += 1

                            cur_speaker = "writer"
                            iu_context = f"### Text Before Edit:\n{cur_sent1}\n\n### Text After Edit:\n{cur_sent2}"

                            cur_edit_type_str = str(edit_dict["type"]).lower().strip()
                            assert cur_edit_type_str in edit_types
                            assert "sentence-1-token-indices" in edit_dict and "sentence-2-token-indices" in edit_dict
                            sent1_idx = edit_dict["sentence-1-token-indices"]
                            sent2_idx = edit_dict["sentence-2-token-indices"]
                            if cur_edit_type_str == "insertion":
                                assert isinstance(sent2_idx, list) and len(sent2_idx) == 2
                                assert 0 <= sent2_idx[0] < sent2_idx[1] <= len(cur_sent2_tokens)
                                cur_target = cur_sent2_tokens[sent2_idx[0]: sent2_idx[1]]
                                cur_target_str = " ".join(cur_target)
                                cur_text_raw = f"Edit: inserting \"{cur_target_str}\" at word-index {sent2_idx[0]}."
                                iu_question_raw = (f"The {cur_speaker} inserted \"{cur_target_str}\" at "
                                                   f"word-index {sent2_idx[0]}. "
                                                   f"What is the intent of the {cur_speaker} for the edit?")
                            elif cur_edit_type_str == "deletion":
                                assert isinstance(sent1_idx, list) and len(sent1_idx) == 2
                                assert 0 <= sent1_idx[0] < sent1_idx[1] <= len(cur_sent1_tokens)
                                cur_target = cur_sent1_tokens[sent1_idx[0]: sent1_idx[1]]
                                cur_target_str = " ".join(cur_target)
                                cur_text_raw = f"Edit: deleting \"{cur_target_str}\" at word-index {sent1_idx[0]}."
                                iu_question_raw = (f"The {cur_speaker} deleted \"{cur_target_str}\" at "
                                                   f"word-index {sent1_idx[0]}. "
                                                   f"What is the intent of the {cur_speaker} for the edit?")
                            elif cur_edit_type_str == "substitute":
                                assert isinstance(sent1_idx, list) and len(sent1_idx) == 2
                                assert 0 <= sent1_idx[0] < sent1_idx[1] <= len(cur_sent1_tokens)
                                cur_source = cur_sent1_tokens[sent1_idx[0]: sent1_idx[1]]
                                cur_source_str = " ".join(cur_source)
                                assert isinstance(sent2_idx, list) and len(sent2_idx) == 2
                                assert 0 <= sent2_idx[0] < sent2_idx[1] <= len(cur_sent2_tokens)
                                cur_target = cur_sent2_tokens[sent2_idx[0]: sent2_idx[1]]
                                cur_target_str = " ".join(cur_target)
                                cur_text_raw = (f"Edit: substituting \"{cur_source_str}\" "
                                                f"with \"{cur_target_str}\" at word-index {sent1_idx[0]}.")
                                iu_question_raw = (f"The {cur_speaker} substituted \"{cur_source_str}\" with "
                                                   f"\"{cur_target_str}\" at word-index {sent1_idx[0]}. "
                                                   f"What is the intent of the {cur_speaker} for the edit?")
                            else:
                                raise ValueError(f"Unknown edit type: {cur_edit_type_str}")

                            iu_answer_intent_raw = []
                            assert cur_intent_str in self.intent_label2statement
                            iu_answer_intent_raw.append(self.intent_label2statement[cur_intent_str])

                            cur_domain_topic = []
                            assert cur_intent_str in self.intent_label2category
                            cur_domain, cur_topic = self.intent_label2category[cur_intent_str]
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
                                "paper_year": 2022,  # The year of publication/preprint
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
