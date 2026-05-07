# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskBanking77(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "banking77"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2020.nlp4convai-1.5/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }

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
            "card payment fee charged": "To inquire: card payment fee charged.",  # 227,
            "direct debit payment not recognised": "To inquire: direct debit payment not recognised.",  # 222,
            "balance not updated after cheque or cash deposit":
                "To inquire: balance not updated after cheque or cash deposit.",  # 221,
            "wrong amount of cash received": "To inquire: wrong amount of cash received.",  # 220,
            "cash withdrawal charge": "To inquire about the cash withdrawal charge.",  # 217,
            "transaction charged twice": "To inquire: transaction charged twice.",  # 215,
            "declined cash withdrawal": "To inquire about the declined cash withdrawal.",  # 213,
            "transfer fee charged": "To inquire: transfer fee charged.",  # 212,
            "transfer not received by recipient": "To inquire: transfer not received by recipient.",  # 211,
            "balance not updated after bank transfer": "To inquire: balance not updated after bank transfer.",  # 211,
            "request refund": "To request a refund.",  # 209,
            "card payment not recognised": "To inquire: card payment not recognised.",  # 208,
            "card payment wrong exchange rate": "To inquire: card payment wrong exchange rate.",  # 207,
            "extra charge on statement": "To inquire about the extra charge on statement.",  # 206,
            "wrong exchange rate for cash withdrawal":
                "To inquire about the wrong exchange rate for cash withdrawal.",  # 203,
            "refund not showing up": "To inquire: refund not showing up.",  # 202,
            "reverted card payment?": "To inquire about the reverted card payment.",  # 201,
            "cash withdrawal not recognised": "To inquire: cash withdrawal not recognised.",  # 200,
            "pending card payment": "To inquire about the pending card payment.",  # 199,
            "activate my card": "To activate the card.",  # 199,
            "cancel transfer": "To cancel the transfer.",  # 197,
            "beneficiary not allowed": "To inquire: beneficiary not allowed.",  # 196,
            "card arrival": "To inquire about the card arrival.",  # 193,
            "declined card payment": "To inquire about the declined card payment.",  # 193,
            "pending top up": "To inquire about the pending top-up.",  # 189,
            "pending transfer": "To inquire about the pending transfer.",  # 188,
            "top up reverted": "To inquire about the reverted top-up.",  # 186,
            "top up failed": "To inquire about the failed top-up.",  # 185,
            "pending cash withdrawal": "To inquire about the pending case withdrawal.",  # 183,
            "card linking": "To inquire about card linking.",  # 179,
            "failed transfer": "To inquire about the failed transfer.",  # 177,
            "visa or mastercard": "To inquire about visa or mastercard.",  # 175,
            "declined transfer": "To inquire about the declined transfer.",  # 173,
            "supported cards and currencies": "To inquire about the supported cards and currencies.",  # 169,
            "getting spare card": "To inquire about getting a spare card.",  # 169,
            "card about to expire": "To inquire about a card that is about to expire.",  # 169,
            "country support": "To inquire about the country support.",  # 169,
            "transfer timing": "To inquire about the transfer timing.",  # 168,
            "automatic top up": "To inquire about the automatic top-up.",  # 167,
            "fiat currency support": "To inquire about the fiat currency support.",  # 166,
            "verify top up": "To verify the top-up.",  # 166,
            "apple pay or google pay": "To inquire about apple pay or google pay.",  # 166,
            "change pin": "To change the pin.",  # 162,
            "edit personal details": "To edit personal details.",  # 161,
            "why verify identity": "To ask the reasons for verifying identity.",  # 161,
            "disposable card limits": "To inquire about the disposable card limits.",  # 161,
            "lost or stolen phone": "To inquire about the lost or stolen phone.",  # 161,
            "exchange charge": "To inquire about the exchange charge.",  # 161,
            "order physical card": "To order a physical card.",  # 160,
            "exchange via app": "To inquire about exchange via app.",  # 158,
            "pin blocked": "To inquire: pin blocked.",  # 155,
            "top up by cash or cheque": "To inquire about top-up by cash or cheque.",  # 154,
            "top up by card charge": "To inquire about top-up by card charge.",  # 154,
            "verify source of funds": "To verify the source of funds.",  # 153,
            "transfer into account": "To transfer money into the account.",  # 153,
            "exchange rate": "To inquire about the exchange rate.",  # 152,
            "card delivery estimate": "To inquire about the estimated date for card delivery.",  # 152,
            "card not working": "To inquire: card not working.",  # 152,
            "top up by bank transfer charge": "To inquire: top-up by bank transfer charge.",  # 151,
            "age limit": "To inquire about the age limit.",  # 150,
            "terminate account": "To terminate the account.",  # 148,
            "get physical card": "To inquire about getting the PIN.",  # 146,  # "To get a physical card."
            "passcode forgotten": "To inquire: passcode forgotten.",  # 145,
            "verify my identity": "To verify the identity.",  # 144,
            "topping up by card": "To inquire: topping up by card.",  # 143,
            "unable to verify identity": "To inquire: unable to verify identity.",  # 142,
            "getting virtual card": "To get a virtual card.",  # 138,
            "top up limits": "To inquire about the top-up limits.",  # 137,
            "get disposable virtual card": "To get a disposable virtual card.",  # 137,
            "receiving money": "To inquire about receiving money.",  # 135,
            "atm support": "To inquire about the ATM support.",  # 127,
            "compromised card": "To inquire about a compromised card.",  # 126,
            "lost or stolen card": "To inquire about the lost or stolen card.",  # 122,
            "card swallowed": "To inquire: card swallowed.",  # 101,
            "card acceptance": "To inquire about the acceptance scope of the card.",  # 99,
            "virtual card not working": "To inquire: virtual card not working.",  # 81,
            "contactless not working": "To inquire: contactless not working.",  # 75,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            intent_clear = intent_raw.replace("_", " ")

            if intent_clear == "Refund not showing up":
                intent_clear = "refund not showing up"

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
                assert file_format == "csv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = ","
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item[1]).strip()
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
                    cur_text_raw = str(raw_item[0]).strip()
                    cur_intent_label = str(raw_item[1]).strip()

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "costumer"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []
                    cur_domain, cur_topic = "daily life", "banking"
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
