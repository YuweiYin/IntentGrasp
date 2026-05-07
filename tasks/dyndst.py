# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random
import re
import copy

from tasks import TaskManager
from utils.data_io import DataIO


class TaskDynDST(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "dyndst"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": True,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2025,  # The year of publication/preprint
            "paper_url": "https://ojs.aaai.org/index.php/AAAI/article/view/34534",  # URL of the dataset paper
            "license": "Apache",  # the releasing license of the original dataset (ACL publication - CC-BY-NC-SA)
            "intent_description": {},  # The description of each intent label
        }

        self.task_data = {
            "train": {
                "filenames": [
                    "multiwoz_2.2_dynamic_lite_inject_new_correct_length_long_include_update_response_0_include_options_1.pt",
                    # "multiwoz_2.2_dynamic_lite_inject_new_correct_length_short_include_update_response_0_include_options_1.pt",
                    "multiwoz_2.2_dynamic_lite_keic_otc_correct_length_long_include_update_response_1_include_options_1.pt",
                    # "multiwoz_2.2_dynamic_lite_keic_otc_correct_length_short_include_update_response_1_include_options_1.pt",
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": [
                    "multiwoz_2.2_dynamic_lite_inject_new_correct_length_long_include_update_response_0_include_options_1.pt",
                    # "multiwoz_2.2_dynamic_lite_inject_new_correct_length_short_include_update_response_0_include_options_1.pt",
                    "multiwoz_2.2_dynamic_lite_keic_otc_correct_length_long_include_update_response_1_include_options_1.pt",
                    # "multiwoz_2.2_dynamic_lite_keic_otc_correct_length_short_include_update_response_1_include_options_1.pt",
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": [
                    "multiwoz_2.2_dynamic_lite_inject_new_correct_length_long_include_update_response_0_include_options_1.pt",
                    # "multiwoz_2.2_dynamic_lite_inject_new_correct_length_short_include_update_response_0_include_options_1.pt",
                    "multiwoz_2.2_dynamic_lite_keic_otc_correct_length_long_include_update_response_1_include_options_1.pt",
                    # "multiwoz_2.2_dynamic_lite_keic_otc_correct_length_short_include_update_response_1_include_options_1.pt",
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "train-departure": "To inquire about the departure of the train.",  # 316,
            "train-destination": "To inquire about the destination of the train.",  # 288,
            "attraction-type": "To inquire about the tourist attraction type.",  # 280,
            "restaurant-food": "To inquire about restaurant food.",  # 240,
            "train-day": "To inform or inquire about the date for taking the train.",  # 236,
            "train-bookpeople": "To inform or inquire about people who will take the train.",  # 208,
            "restaurant-pricerange": "To inquire about the restaurant price.",  # 208,
            "restaurant-name": "To inquire about the restaurant name.",  # 196,
            "hotel-type": "To inquire about the type of the hotel.",  # 180,
            "hotel-pricerange": "To inquire about the hotel price.",  # 156,
            "attraction-area": "To inquire about the area information of the tourist attraction.",  # 152,
            "attraction-name": "To inquire about the tourist attraction name.",  # 136,
            "restaurant-bookpeople": "To inform or inquire about people who will dine in the restaurant.",  # 128,
            "hotel-bookpeople": "To inform or inquire about people who will stay in the hotel.",  # 124,
            "hotel-stars": "To inquire about hotel stars.",  # 112,
            "train-leaveat": "To ask when the train will leave.",  # 108,
            "taxi-departure": "To inquire about the departure of the taxi.",  # 108,
            "restaurant-area": "To inquire about the area information of the restaurant.",  # 108,
            "hotel-name": "To inquire about the hotel name.",  # 104,
            "hotel-area": "To inquire about the area information of the hotel.",  # 96,
            "taxi-destination": "To inform or inquire about the destination of the taxi.",  # 88,
            "train-arriveby": "To ask when the train will arrive.",  # 84,
            "taxi-leaveat": "To ask when the taxi will leave.",  # 76,
            "hotel-bookstay": "To inform or inquire about the stay in the hotel.",
            "hotel-bookday": "To inform or inquire about the date to stay in the hotel.",
            "restaurant-booktime": "To inform or inquire about the time to dine in the restaurant.",
            "restaurant-bookday": "To inform or inquire about the date to dine in the restaurant.",
            "taxi-arriveby": "To ask when the taxi will arrive.",
            "hospital-department": "To inquire about the hospital.",
            "attraction-entrancefee": "To inquire about the entrance fee of the tourist attraction.",
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "train-departure": ["daily life", "train"],
            "train-destination": ["daily life", "train"],
            "attraction-type": ["daily life", "tourist attraction"],
            "restaurant-food": ["daily life", "restaurant"],
            "train-day": ["daily life", "train"],
            "train-bookpeople": ["daily life", "train"],
            "restaurant-pricerange": ["daily life", "restaurant"],
            "restaurant-name": ["daily life", "restaurant"],
            "hotel-type": ["daily life", "hotel"],
            "hotel-pricerange": ["daily life", "hotel"],
            "attraction-area": ["daily life", "tourist attraction"],
            "attraction-name": ["daily life", "tourist attraction"],
            "restaurant-bookpeople": ["daily life", "restaurant"],
            "hotel-bookpeople": ["daily life", "hotel"],
            "hotel-stars": ["daily life", "hotel"],
            "train-leaveat": ["daily life", "train"],
            "taxi-departure": ["daily life", "taxi"],
            "restaurant-area": ["daily life", "restaurant"],
            "hotel-name": ["daily life", "hotel"],
            "hotel-area": ["daily life", "hotel"],
            "taxi-destination": ["daily life", "taxi"],
            "train-arriveby": ["daily life", "train"],
            "taxi-leaveat": ["daily life", "taxi"],
            "hotel-bookstay": ["daily life", "hotel"],
            "hotel-bookday": ["daily life", "hotel"],
            "restaurant-booktime": ["daily life", "restaurant"],
            "restaurant-bookday": ["daily life", "restaurant"],
            "taxi-arriveby": ["daily life", "taxi"],
            "hospital-department": ["daily life", "hospital"],
            "attraction-entrancefee": ["daily life", "tourist attraction"],
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.lower().strip()
            return [intent_raw]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        all_intent2text = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "pt"
                cur_data = DataIO.load_pt(str(cur_filepath), verbose=self.verbose)
                assert isinstance(cur_data, dict)
                task_split_key = task_split if task_split != "valid" else "validation"
                if task_split_key in cur_data:
                    cur_data = cur_data[task_split_key]
                    assert isinstance(cur_data, dict)
                    assert "inputs" in cur_data and "outputs" in cur_data and "old_answer" in cur_data
                    assert "conversation_flow" in cur_data and "slot_name" in cur_data and "edit_idx" in cur_data
                    cur_data_inputs = cur_data["inputs"]
                    cur_data_outputs = cur_data["outputs"]
                    cur_data_old_answer = cur_data["old_answer"]
                    cur_data_conv_flow = cur_data["conversation_flow"]
                    cur_data_slot_name = cur_data["slot_name"]
                    cur_data_edit_idx = cur_data["edit_idx"]
                    assert isinstance(cur_data_inputs, list) and isinstance(cur_data_outputs, list)
                    assert isinstance(cur_data_old_answer, list) and isinstance(cur_data_conv_flow, list)
                    assert isinstance(cur_data_slot_name, list) and isinstance(cur_data_edit_idx, list)
                    assert (len(cur_data_inputs) == len(cur_data_outputs) == len(cur_data_old_answer) ==
                            len(cur_data_conv_flow) == len(cur_data_slot_name) == len(cur_data_edit_idx) > 0)

                    cur_data_list = []
                    for _i, _o, _a, _c, _s, _e in zip(cur_data_inputs, cur_data_outputs, cur_data_old_answer,
                                                      cur_data_conv_flow, cur_data_slot_name, cur_data_edit_idx):
                        cur_data_item = {
                            "inputs": _i,
                            "outputs": _o,
                            "old_answer": _a,
                            "conversation_flow": _c,
                            "slot_name": _s,
                            "edit_idx": _e,
                        }
                        cur_data_list.append(cur_data_item)
                else:
                    continue
                cur_split_raw_data += cur_data_list

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item["slot_name"]).strip()
                for _cur_intent_label in _normalize_intent_label(cur_intent_label):
                    if _cur_intent_label not in all_intents_dict:
                        all_intents_dict[_cur_intent_label] = 1
                    else:
                        all_intents_dict[_cur_intent_label] += 1

                    cur_text_raw = list(raw_item["inputs"])[0]
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

        skip_cnt = 0

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
                assert file_format == "pt"
                cur_data = DataIO.load_pt(str(cur_filepath), verbose=self.verbose)
                assert isinstance(cur_data, dict)
                task_split_key = task_split if task_split != "valid" else "validation"
                if task_split_key in cur_data:
                    cur_data = cur_data[task_split_key]
                    assert isinstance(cur_data, dict)
                    assert "inputs" in cur_data and "outputs" in cur_data and "old_answer" in cur_data
                    assert "conversation_flow" in cur_data and "slot_name" in cur_data and "edit_idx" in cur_data
                    cur_data_inputs = cur_data["inputs"]
                    cur_data_outputs = cur_data["outputs"]
                    cur_data_old_answer = cur_data["old_answer"]
                    cur_data_conv_flow = cur_data["conversation_flow"]
                    cur_data_slot_name = cur_data["slot_name"]
                    cur_data_edit_idx = cur_data["edit_idx"]
                    assert isinstance(cur_data_inputs, list) and isinstance(cur_data_outputs, list)
                    assert isinstance(cur_data_old_answer, list) and isinstance(cur_data_conv_flow, list)
                    assert isinstance(cur_data_slot_name, list) and isinstance(cur_data_edit_idx, list)
                    assert (len(cur_data_inputs) == len(cur_data_outputs) == len(cur_data_old_answer) ==
                            len(cur_data_conv_flow) == len(cur_data_slot_name) == len(cur_data_edit_idx) > 0)

                    cur_data_list = []
                    for _i, _o, _a, _c, _s, _e in zip(cur_data_inputs, cur_data_outputs, cur_data_old_answer,
                                                      cur_data_conv_flow, cur_data_slot_name, cur_data_edit_idx):
                        cur_data_item = {
                            "inputs": _i,
                            "outputs": _o,
                            "old_answer": _a,
                            "conversation_flow": _c,
                            "slot_name": _s,
                            "edit_idx": _e,
                        }
                        cur_data_list.append(cur_data_item)
                else:
                    continue
                cur_file_raw_data = cur_data_list

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_item_inputs = list(raw_item["inputs"])
                    cur_item_outputs = list(raw_item["outputs"])
                    assert len(cur_item_inputs) == len(cur_item_outputs) > 0

                    cur_intent_label = str(raw_item["slot_name"]).strip()
                    assert cur_intent_label in self.intent_label2statement

                    for cur_item_input, cur_item_output in zip(cur_item_inputs, cur_item_outputs):
                        assert isinstance(cur_item_input, list) and len(cur_item_input) >= 2
                        assert isinstance(cur_item_output, str)
                        cur_item_output = cur_item_output.lower().strip()

                        cur_speaker = "user"
                        all_speakers = ["user", "agent"]
                        cur_speaker_idx = 0
                        cur_dialogue = ""
                        for utterance in cur_item_input[:-1]:
                            utterance = str(utterance).replace("\n", " ").strip()
                            cur_dialogue += f"{all_speakers[cur_speaker_idx]}: {utterance}\n"
                            cur_speaker_idx = (cur_speaker_idx + 1) % len(all_speakers)
                        cur_dialogue = cur_dialogue.strip()
                        assert len(cur_dialogue) > 0

                        cur_q_op = str(cur_item_input[-1]).replace("\n", " ").strip()
                        if "(Options:" not in cur_q_op:
                            skip_cnt += 1
                            continue
                        cur_q = cur_q_op.split("(Options:")[0].strip()
                        cur_q = cur_q.strip()

                        cur_options_find = re.findall(r"\(Options:(.*?)\)", cur_q_op)
                        if len(cur_options_find) != 1:
                            skip_cnt += 1
                            continue
                        cur_options_str = str(cur_options_find[0]).strip()
                        if len(cur_options_str) == 0 or "," not in cur_options_str:
                            skip_cnt += 1
                            continue
                        cur_options = cur_options_str.split(",")
                        cur_options = [str(_op).lower().strip() for _op in cur_options]
                        if cur_item_output not in cur_options:
                            skip_cnt += 1
                            continue

                        iu_answer_index = 0
                        for op_idx, cur_op in enumerate(cur_options):
                            if cur_item_output == cur_op:
                                iu_answer_index = op_idx
                                break

                        iu_context = f"### Conversation:\n{cur_dialogue}"
                        iu_question_raw = (f"Note that the {cur_speaker} changed their intent during the conversation. "
                                           f"Please answer the following question from the {cur_speaker}:\n{cur_q}")

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
                        cur_item = {
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
                            "answer_intent": [cur_item_output],  # intent_statement: List[str]
                            "answer_index": [iu_answer_index],  # List[int] (could have multiple correct intents)
                            "options": cur_options,  # List[str] (including the correct answer)
                        }

                        # Check data types & format
                        assert isinstance(cur_item["paper_year"], int) and cur_item["paper_year"] > 0
                        cur_domain_topic = cur_item["domain_topic"]
                        assert isinstance(cur_domain_topic, list) and len(cur_domain_topic) > 0
                        for d_t in cur_domain_topic:
                            assert isinstance(d_t, list) and len(d_t) == 2
                        cur_answer_intent = cur_item["answer_intent"]  # there must be >= 1 correct intent answer
                        assert isinstance(cur_answer_intent, list) and len(cur_answer_intent) > 0
                        cur_answer_index = cur_item["answer_index"]  # the answer index can 1-to-1 map to answer intent
                        assert isinstance(cur_answer_index, list) and len(cur_answer_index) == len(
                            cur_answer_intent) > 0
                        cur_options = cur_item["options"]  # # of options must be larger than # of correct intents
                        assert isinstance(cur_options, list) and len(cur_options) > len(cur_answer_intent) > 0
                        cur_options_set = set(cur_options)
                        for _intent in cur_answer_intent:  # the correct intent answer must be in the options list
                            assert isinstance(_intent, str) and _intent in cur_options_set

                        # Limit the number of options
                        if len(cur_options) >= self.num_options:
                            cur_options_cut = copy.deepcopy(cur_answer_intent)
                            cur_answer_intent_set = set(cur_answer_intent)
                            for op in cur_options:
                                if op not in cur_answer_intent_set:
                                    cur_options_cut.append(op)
                                if len(cur_options_cut) >= self.num_options:
                                    break
                            cur_options = cur_options_cut
                        assert self.num_options >= len(cur_options) >= len(cur_answer_intent) > 0

                        random.shuffle(cur_options)
                        cur_options_set = set(cur_options)
                        for _intent in cur_answer_intent:  # double-check: the correct intent must be in the options
                            assert isinstance(_intent, str) and _intent in cur_options_set
                        cur_item["options"] = cur_options

                        assert len(cur_item) == 15
                        cur_item_processed = {
                            "id": str(cur_item["id"]).strip(),
                            "metadata": {
                                "id": str(cur_item["id"]).strip(),
                                "paper_year": int(cur_item["paper_year"]),
                                "original_task": str(cur_item["original_task"]).strip(),
                                "original_split": str(cur_item["original_split"]).strip(),
                                "text_form": str(cur_item["text_form"]).strip(),
                                "intent_type": str(cur_item["intent_type"]).strip(),
                                "is_synthetic": bool(cur_item["is_synthetic"]),
                                "is_sensitive": bool(cur_item["is_sensitive"]),
                                "domain_topic": list(cur_item["domain_topic"]),
                            },
                            "speaker": str(cur_item["speaker"]).strip(),
                            "context": str(cur_item["context"]).strip(),
                            "question": str(cur_item["question"]).strip(),
                            "options": cur_item["options"],
                            "answer_intent": cur_item["answer_intent"],
                            "answer_index": cur_item["answer_index"],
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
