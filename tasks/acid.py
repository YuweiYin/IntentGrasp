# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskACID(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "acid"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://ceur-ws.org/Vol-2666/KDD_Converse20_paper_10.pdf",  # URL of the dataset paper
            # "The intent prediction dataset was collected from past interaction of customers
            #   with our service representatives at American Family Insurance."
            "license": "CC-BY",  # the releasing license of the original dataset (KDD publication - CC-BY)
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
            "add/remove vehicle": "To add or remove vehicle.",  # 378,
            "login error": "To report login errors.",  # 371,
            "emergency roadside service": "To inquire about ",  # 359,
            "add/remove insuranceured": "To add or remove insurance.",  # 358,
            "careers": "To ask for career advice.",  # 324,
            "different amounts": "To inquire about different amounts.",  # 321,
            "speak to representatives": "To speak to the representatives.",  # 310,
            "cancel insurance policy": "To cancel insurance policy.",  # 309,
            "update lienholder": "To update the lienholder information.",  # 305,
            "delete double payment": "To delete the double payment.",  # 295,
            "can not see farm ranch policy": "To report an issue: can not see farm ranch policy.",  # 289,
            "auto insurance canada": "To inquire about the auto insurance in Canada.",  # 286,
            "declaration page needed": "To inquire about the declaration page.",  # 283,
            "life beneficiary change": "To inquire about changing the life beneficiary.",  # 281,
            "make payments": "To make payments.",  # 278,
            "credit card fee": "To inquire about the credit card fee.",  # 271,
            "glass coverage": "To inquire about the glass coverage.",  # 270,
            "policy doc needed": "To inquire about the policy documents.",  # 267,
            "agent not responding": "To report an issue: agent not responding.",  # 265,
            "auto insurance coverage question": "To inquire about the auto insurance coverage.",  # 263,
            "insurance card proof": "To inquire about the insurance card proof.",  # 259,
            "add vehicle property paperless billing": "To add the vehicle property paperless billing.",  # 258,
            "business policy can not see": "To report an issue: can not see the business policy.",  # 258,
            "bill due date": "To inquire about the bill due date.",  # 252,
            "discounts": "To inquire about the discounts.",  # 252,
            "credit card change num": "To change the credit card number.",  # 251,
            "general policy coverage question": "To inquire about the policy coverage.",  # 251,
            "life policy cannot see": "To report an issue: can not see the life policy.",  # 250,
            # "claim: status": "To check the status.",  # 249,
            "refund check": "To check the refund.",  # 237,
            "update phone number": "To update the phone number.",  # 237,
            "get a quote": "To get a quote.",  # 236,
            "health insurance quote": "To inquire about the health insurance quote.",  # 232,
            "amount due": "To inquire about the amount due.",  # 231,
            "combine payments": "To inquire about the combine payments.",  # 231,
            # "claim: adjuster info": "To get the adjuster information.",  # 228,
            "reinstate insurance policy": "To reinstate the insurance policy.",  # 227,
            "automatic payment schedule": "To inquire about the automatic payment schedule.",  # 223,
            "payment not ontime": "To report an issue: the payment is not on time.",  # 223,
            "life cash out": "To inquire about cashing out the life insurance.",  # 222,
            "why was policy cancelled": "To ask the reasons for cancelling the policy.",  # 221,
            "find agent": "To find an agent.",  # 220,
            "automatic payment cancel": "To cancel the automatic payment.",  # 218,
            # "claim: file a claim": "To file a claim.",  # 215,
            "operating area": "To inquire about the operating area.",  # 213,
            "payment duedate change": "To change the payment due date.",  # 213,
            "insurance not available": "To report an issue: insurance not available.",  # 212,
            "one time payment": "To inquire about the one-time payment.",  # 207,
            "increase life insurance": "To increase the life insurance.",  # 206,
            "policy number": "To inquire about the policy number.",  # 205,
            "billing account num": "To inquire about the billing account number.",  # 203,
            "change user id": "To change the user id.",  # 203,
            "transfer account balance": "To transfer the account balance.",  # 201,
            "paperless docs stop": "To stop the paperless billing.",  # 200,
            "paperless docs setup": "To set up the paperless billing.",  # 199,
            "payment history": "To inquire about the payment history.",  # 198,
            "name change": "To change the name.",  # 197,
            "set up account": "To set up the account.",  # 192,
            "life refund": "To inquire about the life insurance refund.",  # 191,
            "mexico auto insurance": "To inquire about Mexico auto insurance.",  # 191,
            "billing department contact": "To inquire about the billing department contact.",  # 189,
            "change bank account": "To change the bank account.",  # 189,
            "forgot user id": "To report an issue: forgot user id.",  # 184,
            "pay life insurance bills": "To pay the life insurance bills.",  # 184,
            "payment confirm": "To confirm the payment.",  # 183,
            # "claim: docs email": "To email the docs of the claim.",  # 180,
            "update email": "To update the email.",  # 179,
            "mailing address for payment": "To inquire about the mailing address for payment.",  # 178,
            "update contact info": "To update the contact information.",  # 178,
            "prepaid card payment": "To inquire about the prepaid card payment.",  # 177,
            "can not see policy": "To report an issue: can not see the policy.",  # 174,
            "policy trans to rental": "To inquire about the policy trans to rental.",  # 173,
            "handling fee remove": "To remove the handling fee.",  # 172,
            "rideshare coverage": "To inquire about the ride-share coverage.",  # 170,
            # "claim: claim filed": "To inquire about why the claim failed.",  # 169,
            "forgot password": "To report an issue: forgot password.",  # 168,
            # "claim: complaint": "To claim a complaint.",  # 167,
            "payment process change": "To change the payment process.",  # 163,
            "cancel fee": "To inquire about the cancellation fee.",  # 162,
            "life update contact info": "To update the contact information for life insurance.",  # 160,
            "life policy cancel": "To cancel the life insurance policy.",  # 156,
            "life policy amount due": "To inquire about the life insurance policy amount due.",  # 155,
            # "claim: docs fax": "",  # 149,
            "get a business insurance quote": "To get a business insurance quote.",  # 146,
            "payment setup automatic payment": "To set up the automatic payment.",  # 140,
            "phone number international": "To inquire about the international phone number.",  # 122,
            "letter of experience": "To inquire about the letter of experience.",  # 119,
            "life policy automatic payment": "To inquire about the life insurance policy automatic payment.",  # 119,
            "auto insurance policy can not see in account":
                "To report an issue: can not see the auto insurance policy in the account.",  # 118,
            # "claim: update info": "To update the information.",  # 117,
            "forgot email": "To report an issue: forgot email.",  # 116,
            "log out": "To log out.",  # 100,
            "do not contact": "To ask the other side not to contact me anymore.",  # 99,
            "deductible": "To ask how much is deductible for the insurance.",  # 96,
            "get a auto insurance quote": "To get an auto insurance quote.",  # 94,
            # "claim: file claim": "To file a claim.",  # 92,
            "collections": "To inquire about the collections.",  # 90,
            "paperless mail": "To inquire about the paperless mail.",  # 89,
            # "claim: docs mail": "",  # 71,
            "cancel confirm": "To confirm the cancellation.",  # 68,
            "phone number": "To inquire about the phone number.",  # 65,
            "operating company": "To inquire about the operating company.",  # 63,
            "payment error": "To inquire about payment errors.",  # 61,
            "premium breakdown": "To inquire about the premium breakdown.",  # 41,
            # "ast quote": "",  # 37,
            # "general request": "",  # 37,
            "emergency roadside service contact": "To contact the emergency roadside service.",  # 35,
            "change agent": "To change the agent.",  # 30,
            # "claim: docs send": "",  # 29,
            # "claim: check status": "",  # 28,
            "insurance card print": "To print the insurance card.",  # 28,
            "confirm coverageerage": "To confirm the coverage status.",  # 27,
            # "claim: rental": "",  # 25,
            "insurance card send": "To send the insurance card.",  # 25,
            "who is my agent": "To ask who the agent is.",  # 24,
            # "thank you": "",  # 24,
            # "mortgage co poi": "",  # 22,
            "get a quote renters": "To get a quote on rental insurance.",  # 21,
            "new vehicle grace period": "To inquire about the new vehicle grace period.",  # 21,
            # "no": "",  # 19,
            "gap insurance coverage": "To inquire about the gap insurance coverage.",  # 18,
            "change autopay date": "To change the auto pay date.",  # 17,
            # "claim: drp assign": "",  # 17,
            "dreams foundation": "To inquire about the Dreams foundation.",  # 17,
            "poi old": "To inquire about the old policy or proof of insurance.",  # 17,
            # "ast purchase": "",  # 16,
            # "claim: shop add work": "",  # 16,
            # "claim: shop send estimate": "",  # 16,
            # "life question general": "",  # 16,
            # "srtwentytwo": "",  # 16,
            # "yes": "",  # 16,
            # "claim: file a claim auto hail": "",  # 15,
            "employment verify": "To verify the employment information.",  # 15,
            # "the general contact": "",  # 15,
            # "hello": "",  # 15,
            # "agent wrong": "",  # 14,
            "billing account name edit": "To edit the billing account name.",  # 14,
            "business insurance question general": "To inquire about the business insurance.",  # 14,
            # "claim: glass safelite": "",  # 14,
            "comprehensive coverage explanation": "To inquire about or ask for a comprehensive coverage explanation.",  # 14,
            "customer service hours": "To inquire about the customer service hours.",  # 14,
            "knowyourdrive device activate": "To activate the know-your-drive device.",  # 14,
            "boat coverage explanation": "To inquire about or ask for a boat coverage explanation.",  # 13,
            # "claim: drp join": "",  # 13,
            "get a auto insurance quote nonowner": "To get an auto insurance quote from or for a non-owner.",  # 13,
            "homesite contact": "To inquire about the home-site contact.",  # 13,
            "knowyourdrive": "To inquire about the know-your-drive device.",  # 13,
            "knowyourdrive errors": "To inquire about the know-your-drive errors.",  # 13,
            "salvage vehicle": "To inquire about the salvage vehicle.",  # 13,
            "add house": "To add a house to the insurance or policy.",  # 12,
            "automatic payment minimum balance": "To inquire about the automatic payment minimum balance.",  # 12,
            "ded explan": "To ask for an explanation about the deductible.",  # 12,
            "emergency roadside service reimburse": "To inquire about the emergency roadside service reimburse.",  # 12,
            "dreamkeep rewards errors": "To inquire about the DreamKeep rewards errors.",  # 11,
            "flood insurance explanation": "To inquire about or ask for a flood insurance explanation.",  # 11,
            "rv insurance explanation": "To inquire about or ask for a RV insurance explanation.",  # 11,
            # "claim: hrp join": "",  # 11,
            "american star": "To inquire about the American star.",  # 10,
            "atv insurance explanation": "To inquire about or ask for an ATV insurance explanation.",  # 10,
            "collision coverage explanation": "To inquire about or ask for a collision coverage explanation.",  # 10,
            "dreamkeep rewards": "To inquire about the DreamKeep rewards.",  # 10,
            "get a quote renters purchase": "To purchase the rental insurance.",  # 10,
            "knowyourdrive device return": "To return the know-your-drive device.",  # 10,
            "liability explanation": "To inquire about or ask for a liability explanation.",  # 10,
            "payment time": "To inquire about the payment time.",  # 10,
            "renters coverage explanation": "To inquire about or ask for a renters coverage explanation.",  # 10,
            "teen safe driver signup": "To sign up for the teen safe driver.",  # 10,
            "travel insurance explanation": "To inquire about or ask for a travel insurance explanation.",  # 10,
            "uw alumni discount": "To inquire about the UW alumni discount.",  # 10,
            "profile section": "To inquire about the profile.",  # 10,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            "yes", "no", "hello", "thank you", "srtwentytwo",
            "general request", "mortgage co poi", "life question general", "the general contact", "agent wrong",
            "claim: docs email", "claim: claim filed", "claim: complaint", "claim: docs fax", "claim: update info",
            "claim: file claim", "claim: docs mail", "claim: docs send", "claim: check status", "claim: rental",
            "claim: drp assign", "claim: shop add work", "claim: glass safelite", "claim: drp join", "claim: hrp join",
            "claim: file a claim auto hail", "claim: shop send estimate", "claim: status",
            "claim: adjuster info", "claim: file a claim", "ast quote", "ast purchase",
        }

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()

            # Expand the acronyms in the intent labels
            # intent_raw = intent_raw.replace("INFO_CL", "claim:")
            intent_raw = intent_raw.replace("INFO_CL", "claim:")
            intent_raw = intent_raw.replace("LIAB_EXPLAN", "liability explanation")
            intent_raw = intent_raw.replace("ADD_REMOVE", "add/remove")
            intent_raw = intent_raw.replace("AGT_NOT_RESPONDING", "agent not responding")
            intent_raw = intent_raw.replace("AMT_DUE", "amount due")
            intent_raw = intent_raw.replace("AMTS", "amounts")
            intent_raw = intent_raw.replace("INS_EXPLAN", "insurance explanation")
            intent_raw = intent_raw.replace("INS_POLICY", "insurance policy")
            intent_raw = intent_raw.replace("AUTO_COV", "auto insurance coverage")
            intent_raw = intent_raw.replace("AUTO_INS", "auto insurance")
            intent_raw = intent_raw.replace("AUTO_POLICY", "auto insurance policy")
            intent_raw = intent_raw.replace("QUOTE_AUTO", "auto insurance quote")
            intent_raw = intent_raw.replace("ACCT", "account")
            intent_raw = intent_raw.replace("AUTO_PYMT", "automatic payment")
            intent_raw = intent_raw.replace("DUPE_PYMT", "double payment")
            intent_raw = intent_raw.replace("MIN_BALANCE", "minimum balance")
            intent_raw = intent_raw.replace("ACCT_NUM", "account number")
            intent_raw = intent_raw.replace("DEPT", "department")
            intent_raw = intent_raw.replace("CANT", "can not")
            intent_raw = intent_raw.replace("USERID", "user id")
            intent_raw = intent_raw.replace("FNOL", "file a claim")
            intent_raw = intent_raw.replace("PYMTS", "payments")
            intent_raw = intent_raw.replace("COMP_COV_EXPLAN", "comprehensive coverage explanation")
            intent_raw = intent_raw.replace("COLL_COV_EXPLAN", "collision coverage explanation")
            intent_raw = intent_raw.replace("COV_EXPLAN", "coverage explanation")
            intent_raw = intent_raw.replace("DEC_PAGE", "declaration page")
            intent_raw = intent_raw.replace("INFO_ERS", "emergency roadside service")
            intent_raw = intent_raw.replace("GAP_COVERAGE", "gap insurance coverage")
            intent_raw = intent_raw.replace("GEN_POLICY_COV", "general policy coverage")
            intent_raw = intent_raw.replace("QUOTE_CFR", "business insurance quote")
            intent_raw = intent_raw.replace("CFR", "business insurance")
            intent_raw = intent_raw.replace("QUOTE_OTHER", "quote")
            intent_raw = intent_raw.replace("GLASS_COV", "glass coverage")
            intent_raw = intent_raw.replace("INS_QUOTE", "insurance quote")
            intent_raw = intent_raw.replace("INS_NOT_AVAILABLE,", "insurance not available")
            intent_raw = intent_raw.replace("LIFE_INCR_COV", "increase life insurance")
            intent_raw = intent_raw.replace("MAIL_PYMT_ADDRESS", "mailing address for payment")
            intent_raw = intent_raw.replace("MAKE_PYMT", "make payments")
            intent_raw = intent_raw.replace("ONE_TIME_PYMT", "one time payment")
            intent_raw = intent_raw.replace("OPERATING_CO", "operating company")
            intent_raw = intent_raw.replace("PAY_LIFE_INS", "pay life insurance bills")
            intent_raw = intent_raw.replace("PHONE_NUM", "phone number")
            intent_raw = intent_raw.replace("POLICY_NUM", "policy number")
            intent_raw = intent_raw.replace("CARD_PYMT", "card payment")
            intent_raw = intent_raw.replace("PYMT_ERROR", "payment error")
            intent_raw = intent_raw.replace("PYMT_HISTORY", "payment history")
            intent_raw = intent_raw.replace("RV_INS", "RV insurance")
            intent_raw = intent_raw.replace("SPEAK_TO_REP", "speak to representatives")
            intent_raw = intent_raw.replace("_PYMT", "_payment")
            intent_raw = intent_raw.replace("_INS", "_insurance")
            intent_raw = intent_raw.replace("_COV", "_coverage")

            if intent_raw.startswith("INFO_"):
                intent_raw = intent_raw[len("INFO_"):].strip()
            elif intent_raw.startswith("ST_"):
                intent_raw = intent_raw[len("ST_"):].strip()
            else:
                pass
            intent_clear = intent_raw.replace("_", " ").lower()
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
                # csv_header = cur_data[0]
                cur_data = cur_data[1:]  # ignore the csv header
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item[0]).strip()
                if "ABBY" in cur_intent_label:
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
                    cur_text_raw = str(raw_item[1]).strip()
                    cur_intent_label = str(raw_item[0]).strip()
                    if "ABBY" in cur_intent_label:
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

                    cur_domain, cur_topic = "daily life", "insurance"
                    cur_domain_topic = [[cur_domain, cur_topic]]

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
