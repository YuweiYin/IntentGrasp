# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskDSTC11T2(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "dstc11_t2"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2023,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2023.dstc-1.27/",  # URL of the dataset paper
            "license": "Apache",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 4: Data
        #   The process of annotating conversations with reference intents was decoupled from the collection of
        #     conversations in order to mimic the manual process of designing an intent schema based on conversations.
        #   Annotators shared an open intent label set that was periodically reviewed throughout
        #     the process to merge duplicate intents.

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
                "filenames": ["test_banking", "test_finance"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "check account balance": "To check the account balance.",  # 434,
            "apply loan": "To apply for the loan.",  # 222,
            "find branch": "To find the branch.",  # 139,
            "internal funds transfer": "To transfer funds internally.",  # 139,
            "external wire transfer": "To do a wire transfer externally.",  # 124,
            "get loan info": "To get the load information.",  # 111,
            "dispute charge": "To make or inquire about a dispute on a charge.",  # 108,
            "update street address": "To update the street address.",  # 101,
            "get branch hours": "To ask about the branch hours.",  # 98,
            "open banking account": "To open a bank account.",  # 92,
            "find atm": "To find an ATM.",  # 92,
            "report lost stolen card": "To report the lost or stolen card.",  # 77,
            "make transfer": "To make a transfer.",  # 77,
            "update email": "To update the email information.",  # 74,
            "online banking info": "To inquire about the online banking information.",  # 73,
            "update phone number": "To update the phone number.",  # 66,
            "close bank account": "To close the bank account.",  # 65,
            "get credit card info": "To get the credit card information.",  # 55,
            "schedule appointment": "To schedule an appointment.",  # 46,
            # "open account": "To open a bank account.",  # 43,
            "get exchange rate": "To get the exchange rate.",  # 43,
            "change statement delivery": "To change the delivery method for statements.",  # 41,
            "check loan balance": "To check the load balance.",  # 40,
            "apply credit card": "To open or apply for the credit card.",  # 39,
            "change pin": "To change the PIN.",  # 36,
            "request email": "To request information or documents through email.",  # 35,
            "set auto payment": "To set the auto payment.",  # 33,
            "make credit card payment": "To make a credit card payment.",  # 33,
            "request new card": "To request a new card.",  # 32,
            "cancel check": "To cancel the check.",  # 32,
            "get debt income ratio": "To get the debt income ratio.",  # 31,
            "add user to account": "To add a user to the account.",  # 31,
            "ask consumer price index": "To inquire about the consumer price index.",  # 29,
            "close account": "To close an account.",  # 28,
            "check transaction history": "To check the transaction history.",  # 25,
            "ask about transfer time": "To ask about the transfer time.",  # 24,
            "net income": "To inquire about the net income.",  # 24,
            "set up online banking": "To set up online banking.",  # 23,
            "report notice": "To inquire about a notice.",  # 23,
            "order check": "To order checks.",  # 23,
            "get branch info": "To get the branch information.",  # 22,
            "get withdrawal limit": "To get the withdrawal limit.",  # 21,
            "ask liquidity ratio": "To inquire about the liquidity ratio.",  # 21,
            "get bank statement": "To get the bank statement.",  # 17,
            "pay loan": "To pay the loan.",  # 16,
            "ask about cash deposits": "To inquire about cash deposits.",  # 15,
            "get account info": "To get the account information.",  # 14,
            "get transactions": "To get the transactions.",  # 12,
            "check account interest rate": "To check the account interest rate.",  # 11,
            "ask about transfer fees": "To inquire about the transfer fees.",  # 11,
            "ask about card arrival": "To ask when the card will arrive.",  # 10,
            # "order checks": "To order checks.",  # 10,
            # "open credit card": "To open or apply for the credit card.",  # 8,
            "ask about atm fees": "To inquire about ATM fees.",  # 8,
            "get payment due date": "To get the payment due date.",  # 8,
            "get stock quote": "To get the stock quote.",  # 7,
            "get investment report": "To get the investment report.",  # 7,
            "get credit report": "To get the credit report.",  # 6,
            # "get treasury bond yield": "",  # 6,
            "purchase stocks": "To purchase stocks.",  # 5,
            "ask about credit score": "To inquire about the credit score.",  # 4,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "get treasury bond yield",
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
            intent_clear = intent_clear.replace("-", " ").strip()
            intent_clear = intent_clear.replace("  ", " ").strip()
            intent_clear = intent_clear.replace("a t m", "atm").strip()

            if intent_clear == "open account":
                intent_clear = "open banking account"
            if intent_clear == "order checks":
                intent_clear = "order check"
            if intent_clear == "open credit card":
                intent_clear = "apply credit card"

            return [intent_clear.lower().strip()]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename, "dialogues.jsonl")
                assert os.path.isfile(cur_filepath)
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                cur_split_raw_data += cur_data  # Note: each item is a multi-turn dialogue

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                for turn in raw_item["turns"]:
                    if not isinstance(turn, dict) or "intents" not in turn:
                        continue
                    if "speaker_role" not in turn or "utterance" not in turn:
                        continue
                    cur_intents = turn["intents"]
                    if not isinstance(cur_intents, list) or len(cur_intents) == 0:
                        continue
                    _cur_intent_labels = [_normalize_intent_label(_i)[0] for _i in cur_intents]
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
            show_cnt = int(1e5)

            for cur_filename in cur_filenames:
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename, "dialogues.jsonl")
                assert os.path.isfile(cur_filepath)
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                cur_file_raw_data = cur_data  # Note: each item is a multi-turn dialogue

                if "banking" in cur_filename:
                    cur_topic = "banking"
                elif "finance" in cur_filename:
                    cur_topic = "finance"
                else:
                    raise ValueError(f"Unknown filename: {cur_filename}")

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    dialogue_history = ""

                    # cur_num_turns = len(raw_item["turns"])
                    for turn_idx, turn in enumerate(raw_item["turns"]):
                        cur_speaker_raw = str(turn["speaker_role"]).strip()
                        if cur_speaker_raw.lower() == "customer":
                            cur_speaker_raw = "customer"
                        elif cur_speaker_raw.lower() == "agent":
                            cur_speaker_raw = "agent"
                        else:
                            raise ValueError(f"Unknown speaker {cur_speaker_raw}")
                        cur_text_raw = f"{cur_speaker_raw}: " + str(turn["utterance"]).strip()

                        cur_context_raw = dialogue_history
                        dialogue_history = f"{cur_context_raw}\n{cur_text_raw}".strip()

                        if not isinstance(turn, dict) or "intents" not in turn:
                            continue
                        if "speaker_role" not in turn or "utterance" not in turn:
                            continue
                        cur_intents = turn["intents"]
                        if not isinstance(cur_intents, list) or len(cur_intents) == 0:
                            continue
                        _cur_intent_labels = [_normalize_intent_label(_i)[0] for _i in cur_intents]
                        _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                        if len(_cur_intent_labels) == 0:
                            continue
                        for _cur_intent_label in _cur_intent_labels:
                            assert _cur_intent_label in cur_split_intents
                            cur_split_intents[_cur_intent_label] += 1

                        assert cur_speaker_raw == "customer"
                        cur_speaker = cur_speaker_raw
                        iu_context = dialogue_history
                        iu_question_raw = f"What is the intent of the {cur_speaker}?"
                        iu_answer_intent_raw = []
                        for _cur_intent_label in _cur_intent_labels:
                            assert _cur_intent_label in self.intent_label2statement
                            iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                        cur_domain_topic = []
                        cur_domain = "daily life"
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
                            "paper_year": 2023,  # The year of publication/preprint
                            "original_task": self.task_name,  # str
                            "original_split": task_split,  # str
                            "text_form": "dialogue",  # str: query/dialogue/monologue
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
                        if item_idx % show_cnt == 0:
                            self.logger.info(f">>> [{task_split}] Processed items: {item_idx}")

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
