# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskHint3(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "hint3"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2020.insights-1.16/",  # URL of the dataset paper
            "license": "ODbL",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 3.2: Test Data Collection and Annotation
        #   Inter-annotator agreement was 75.8%, 80.0% and 73.4% for SOFMattress, Curekart and Powerplay11
        #     respectively and conflicts were resolved by domain experts.

        self.task_data = {
            "train": {
                "filenames": ["train_sofmattress.csv", "train_curekart.csv", "train_powerplay11.csv"],
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
                "filenames": ["test_sofmattress.csv", "test_curekart.csv", "test_powerplay11.csv"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "recommend product": "To recommend a product.",  # 300,
            "order status": "To check the order status.",  # 88,
            "chat with an agent": "To chat with an agent.",  # 80,
            "refunds returns replacements": "To inquire about refunds, returns, or replacements.",  # 74,
            "check pincode": "To check the pincode.",  # 71,
            "original product": "To inquire about the original product.",  # 66,
            "cancel order": "To cancel the order.",  # 60,
            "resume delivery": "To resume delivery.",  # 60,
            "order taking": "To inquire about taking the order.",  # 58,
            # "chat with agent": "To chat with an agent.",  # 54,
            # "winnings": "",  # 50,
            "points not updated": "To inquire about an issue: points not updated.",  # 50,
            "distributors": "To inquire about the distributors.",  # 47,
            "mattress cost": "To inquire about the mattress cost.",  # 45,
            "what size to order": "To ask what size to order.",  # 42,
            "consult start": "To start the consultation.",  # 42,
            "emi": "To inquire about Equated Monthly Installment (EMI).",  # 38,
            # "lead gen": "",  # 38,
            "delay in parcel": "To inquire about the delay in parcel.",  # 33,
            "comparison": "To ask for comparing products.",  # 29,
            "product variants": "To inquire about the product variants.",  # 29,
            "payment and bill": "To inquire about the payment and bill.",  # 29,
            "check deposit status": "To check the deposit status.",  # 29,
            "ortho features": "To inquire about the features of the Ortho mattress.",  # 28,
            "offers and referrals": "To inquire about the offers and referrals.",  # 28,
            "delay in delivery": "To inquire about the delay in delivery.",  # 26,
            "return exchange": "To inquire about refund, return, or exchange.",  # 26,
            "call center": "To inquire about the call center.",  # 26,
            "store information": "To inquire about the store information.",  # 25,
            "cod": "To inquire about the COD option.",  # 24,
            "size customization": "To customize the size.",  # 24,
            "franchise": "To inquire about the franchise.",  # 24,
            "100 night trial offer": "To inquire about the 100-night trial offer.",  # 23,
            "pillows": "To inquire about the pillows.",  # 23,
            "withdrawal status": "To inquire about the withdrawal status.",  # 23,
            "contact number": "To ask about the contact number.",  # 23,
            # "wrong scores": "To inquire about an issue: wrong score.",  # 22,
            "offers": "To inquire about the offers.",  # 21,
            "expiry date": "To inquire about the expiry date.",  # 21,
            # "fake teams": "",  # 21,
            # "team deadline": "",  # 20,
            "modify address": "To modify the address.",  # 19,
            "account balance deducted": "To inquire about the deduction of account balance.",  # 19,
            "check wallet balance": "To check the wallet balance.",  # 19,
            "withdrawal intro": "To ask for the withdrawal introduction.",  # 18,
            "instant withdrawal": "To inquire about instant withdrawal.",  # 17,
            # "capabilities": "",  # 17,
            # "greetings day": "",  # 17,
            "change mobile number": "To change the mobile number.",  # 15,
            "verify email": "To verify the email.",  # 15,
            "verify pan": "To verify the PAN card.",  # 15,
            "ergo features": "To inquire about the features of the Ergo mattress.",  # 14,
            "about sof mattress": "To inquire about the features of the SOF mattress.",  # 14,
            "user goal form": "To inquire about the user goal form.",  # 14,
            # "change profile team details": "To change the profile team details.",  # 14,
            "cannot see joined contests": "To inquire about an issue: cannot see joined contests.",  # 14,
            # "thanks": "",  # 14,
            "refer earn": "To inquire about the referral.",  # 13,
            "withdrawal time": "To inquire about the withdrawal time.",  # 13,
            # "fairplay violations": "To inquire about FairPlay violations.",  # 12,
            "account not verified": "To inquire about an issue: account not verified.",  # 11,
            "what if theres a tie": "To ask what if there is a tie.",  # 11,
            "warranty": "To inquire about the warranty.",  # 10,
            "sign up": "To sign up.",  # 10,
            # "work from home": "To inquire about the working office information.",  # 10,
            # "join contest": "",  # 10,
            "refund of added cash": "To inquire about the refund of added cash.",  # 10,
            "taxes on winnings": "To inquire about the taxes on winnings.",  # 10,
            "delete pan card": "To delete the PAN card.",  # 10,
            # "new team pattern": "To ask about the new team pattern.",  # 10,
            # "appreciation": "",  # 10,
            # "match abandoned": "",  # 10,
            "side effect": "To inquire about the side effect.",  # 9,
            "verify mobile": "To verify the mobile number.",  # 9,
            # "criticism": "",  # 9,
            "refund of wrong amount": "To inquire about the refund of wrong amount.",  # 8,
            # "less winnings amount": "",  # 8,
            "types bonus": "To inquire about the bonus.",  # 8,
            "how to play": "To ask how to play.",  # 8,
            "modes of payments": "To inquire about the modes of payments.",  # 7,
            "order query": "To inquire about the order.",  # 7,
            "no email confirmation": "To inquire about an issue: no email confirmation.",  # 7,
            "why verify": "To ask the reasons for verification.",  # 7,
            "deducted amount not received": "To inquire about an issue: deducted amount not received.",  # 7,
            "immunity": "To inquire about the immunity.",  # 6,
            "start over": "To start over.",  # 5,
            "change bank account": "To change the bank account.",  # 5,
            # "types contests": "",  # 5,
            "account reset": "To reset the account.",  # 5,
            "portal issue": "To report a portal issue.",  # 4,
            "pan verification failed": "To inquire about an issue: PAN verification failed.",  # 4,
            "how points calculated": "To ask how points are calculated.",  # 4,
            "cash bonus": "To inquire about the cash bonus.",  # 4,
            # "presence": "",  # 4,
            # "download powerplay11": "To download PowerPlay11.",  # 4,
            "international shipping": "To inquire about the international shipping.",  # 3,
            "bank verification details": "To inquire about the bank verification details.",  # 3,
            # "unutilized money": "",  # 3,
            # "feedback": "",  # 3,
            "withdraw cash bonus": "To inquire about the withdraw cash bonus.",  # 2,
            "cash bonus expiry": "To inquire about the cash bonus expiry.",  # 2,
            "signup bonus": "To inquire about the sign-up bonus.",  # 2,
            "update app": "To update the APP.",  # 2,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "thanks", "appreciation", "capabilities", "greetings day", "criticism",
            "feedback", "lead gen", "work from home", "unutilized money",
            # Online Gaming
            "winnings", "wrong scores", "fake teams", "team deadline", "change profile team details",
            "new team pattern", "fairplay violations", "join contest", "match abandoned", "less winnings amount",
            "types contests", "presence", "download powerplay11",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            intent_raw = intent_raw.replace("_", " ").strip()
            intent_raw = intent_raw.lower()

            if intent_raw == "chat with agent":
                intent_raw = "chat with an agent"

            return [intent_raw]

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
                if cur_intent_label == "NO_NODES_DETECTED":
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
                assert file_format == "csv"
                if "table_delimiter" in self.task_data and isinstance(self.task_data["table_delimiter"], str):
                    table_delimiter = self.task_data["table_delimiter"]
                else:
                    table_delimiter = ","
                cur_data = DataIO.load_csv(str(cur_filepath), delimiter=table_delimiter)
                cur_data = cur_data[1:]  # ignore the csv header

                if "sofmattress" in cur_filename:
                    cur_topic = "mattress products retail"
                elif "curekart" in cur_filename:
                    cur_topic = "fitness supplements retail"
                elif "powerplay11" in cur_filename:
                    cur_topic = "online gaming"
                else:
                    raise ValueError(f"Unsupported file: {cur_filename}")
                cur_data = [_d + [cur_topic] for _d in cur_data]

                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item[0]).strip()
                    cur_intent_label = str(raw_item[1]).strip()
                    if cur_intent_label == "NO_NODES_DETECTED":
                        continue

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "user"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                    cur_domain_topic = []
                    cur_domain, cur_topic = "e-commerce", "retail"
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
