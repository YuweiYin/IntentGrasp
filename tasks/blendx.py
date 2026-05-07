# -*- coding: utf-8 -*-

import os
import json
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskBlendX(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "blendx"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": True,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2024,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2024.lrec-main.218/",  # URL of the dataset paper
            "license": "GPL",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 3.4: Dataset Details
        #   We develop the final version of BlendX using both the Manual and Generative concatenation approaches.
        #   We broaden the research scope by incorporating datasets such as Banking77 and CLINC150 and
        #     by utilizing diverse conjunctions.

        self.task_data = {
            "train": {
                "filenames": [
                    "BlendATIS.jsonl", "BlendBanking77.jsonl", "BlendCLINC150.jsonl", "BlendSNIPS.jsonl"
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": [
                    "BlendATIS.jsonl", "BlendBanking77.jsonl", "BlendCLINC150.jsonl", "BlendSNIPS.jsonl"
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": [
                    "BlendATIS.jsonl", "BlendBanking77.jsonl", "BlendCLINC150.jsonl", "BlendSNIPS.jsonl"
                ],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "play music": "",  # 19009,
            "add to playlist": "",  # 18404,
            "search creative work": "",  # 15597,
            "rate book": "",  # 14497,
            "book restaurant": "",  # 14060,
            "search screening event": "",  # 13923,
            "get weather": "",  # 13693,
            "atis flight": "",  # 4670,
            "atis airfare": "",  # 3594,
            "atis airline": "",  # 3358,
            "atis flight time": "",  # 2609,
            "atis aircraft": "",  # 2520,
            "atis ground service": "",  # 2354,
            "atis ground fare": "",  # 2349,
            "atis airport": "",  # 2332,
            "atis abbreviation": "",  # 2303,
            "atis city": "",  # 2281,
            "atis quantity": "",  # 2262,
            "atis distance": "",  # 2256,
            "atis capacity": "",  # 2225,
            "atis meal": "",  # 2194,
            "atis flight no": "",  # 2171,
            "atis restriction": "",  # 2111,
            "atis cheapest": "",  # 1988,
            "pto balance": "",  # 1905,
            "pto used": "",  # 1826,
            "declined card payment": "",  # 1743,
            "reverted card payment?": "",  # 1690,
            "declined cash withdrawal": "",  # 1567,
            "exchange rate": "",  # 1527,
            "why verify identity": "",  # 1493,
            "card payment not recognised": "",  # 1425,
            "card payment fee charged": "",  # 1418,
            "verify my identity": "",  # 1410,
            "wrong amount of cash received": "",  # 1383,
            "cash withdrawal not recognised": "",  # 1357,
            "unable to verify identity": "",  # 1346,
            "pending card payment": "",  # 1344,
            "pending cash withdrawal": "",  # 1292,
            "activate my card": "",  # 1288,
            "card payment wrong exchange rate": "",  # 1283,
            "card arrival": "",  # 1278,
            "wrong exchange rate for cash withdrawal": "",  # 1266,
            "cash withdrawal charge": "",  # 1265,
            "order": "",  # 1261,
            "top up failed": "",  # 1237,
            "shopping list": "",  # 1195,
            "transfer fee charged": "",  # 1179,
            "request refund": "",  # 1172,
            "direct debit payment not recognised": "",  # 1172,
            "extra charge on statement": "",  # 1158,
            "transfer not received by recipient": "",  # 1156,
            "refund not showing up": "",  # 1147,
            "pto request": "",  # 1145,
            "oil change when": "",  # 1144,
            "transaction charged twice": "",  # 1139,
            "pto request status": "",  # 1138,
            "balance not updated after cheque or cash deposit": "",  # 1135,
            "card linking": "",  # 1125,
            "pending top up": "",  # 1104,
            "beneficiary not allowed": "",  # 1100,
            "top up reverted": "",  # 1095,
            "report fraud": "",  # 1095,
            "oil change how": "",  # 1091,
            "card not working": "",  # 1082,
            "pending transfer": "",  # 1063,
            "balance not updated after bank transfer": "",  # 1062,
            "report lost card": "",  # 1061,
            "order physical card": "",  # 1049,
            "cancel transfer": "",  # 1011,
            "declined transfer": "",  # 991,
            "getting spare card": "",  # 977,
            "expiration date": "",  # 975,
            "min payment": "",  # 971,
            "meeting schedule": "",  # 965,
            "credit limit": "",  # 964,
            "card delivery estimate": "",  # 963,
            "balance": "",  # 960,
            "failed transfer": "",  # 950,
            "replacement card duration": "",  # 943,
            "visa or mastercard": "",  # 942,
            "reminder": "",  # 941,
            "application status": "",  # 938,
            "supported cards and currencies": "",  # 936,
            "insurance change": "",  # 936,
            "country support": "",  # 925,
            "insurance": "",  # 923,
            "card about to expire": "",  # 922,
            "schedule meeting": "",  # 921,
            "automatic top up": "",  # 912,
            "credit score": "",  # 911,
            "todo list": "",  # 904,
            "todo list update": "",  # 903,
            "transfer timing": "",  # 902,
            "verify top up": "",  # 901,
            # "change ai name": "",  # 896,
            "fiat currency support": "",  # 894,
            # "what can i ask you": "",  # 883,
            "what is your name": "",  # 881,
            "reminder update": "",  # 881,
            "pin blocked": "",  # 876,
            "apple pay or google pay": "",  # 875,
            "credit limit change": "",  # 871,
            "exchange via app": "",  # 870,
            "change pin": "",  # 867,
            "disposable card limits": "",  # 865,
            "transfer into account": "",  # 859,
            "recipe": "",  # 859,
            "exchange charge": "",  # 857,
            "lost or stolen phone": "",  # 854,
            "improve credit score": "",  # 853,
            "edit personal details": "",  # 852,
            "get physical card": "",  # 850,
            "bill due": "",  # 849,
            "topping up by card": "",  # 846,
            "nutrition info": "",  # 845,
            "top up by bank transfer charge": "",  # 840,
            "new card": "",  # 835,
            "top up by card charge": "",  # 834,
            "restaurant suggestion": "",  # 833,
            "apr": "",  # 832,
            "rewards balance": "",  # 829,
            "confirm reservation": "",  # 825,
            "pay bill": "",  # 823,
            "pin change": "",  # 819,
            "top up by cash or cheque": "",  # 818,
            "ingredients list": "",  # 816,
            "get disposable virtual card": "",  # 804,
            "distance": "",  # 804,
            "verify source of funds": "",  # 802,
            "getting virtual card": "",  # 801,
            "freeze account": "",  # 799,
            "age limit": "",  # 795,
            "card declined": "",  # 793,
            "meal suggestion": "",  # 789,
            "bill balance": "",  # 788,
            "timezone": "",  # 783,
            "greeting": "",  # 782,
            "spending history": "",  # 781,
            "what are your hobbies": "",  # 779,
            "time": "",  # 778,
            # "whisper mode": "",  # 777,
            "restaurant reviews": "",  # 776,
            "roll dice": "",  # 774,
            "travel suggestion": "",  # 772,
            "calendar": "",  # 772,
            "cook time": "",  # 772,
            "damaged card": "",  # 770,
            "directions": "",  # 770,
            "terminate account": "",  # 769,
            "passcode forgotten": "",  # 769,
            # "cancel": "",  # 769,
            "accept reservations": "",  # 769,
            "top up limits": "",  # 768,
            # "change volume": "",  # 767,
            "make call": "",  # 766,
            "redeem rewards": "",  # 765,
            "lost or stolen card": "",  # 762,
            "last maintenance": "",  # 759,
            "alarm": "",  # 758,
            "routing": "",  # 758,
            "cancel reservation": "",  # 757,
            "next holiday": "",  # 757,
            "fun fact": "",  # 755,
            "food last": "",  # 752,
            "next song": "",  # 752,
            # "who made you": "",  # 749,
            "how busy": "",  # 749,
            "transfer": "",  # 747,
            "gas": "",  # 747,
            "taxes": "",  # 746,
            "timer": "",  # 744,
            "update playlist": "",  # 743,
            "account blocked": "",  # 743,
            "book hotel": "",  # 742,
            "international fees": "",  # 741,
            "current location": "",  # 740,
            "tell joke": "",  # 739,
            # "are you a bot": "",  # 737,
            "interest rate": "",  # 737,
            # "repeat": "",  # 736,
            "income": "",  # 735,
            # "reset settings": "",  # 735,
            "transactions": "",  # 735,
            "where are you from": "",  # 730,
            "shopping list update": "",  # 730,
            "date": "",  # 728,
            "travel notification": "",  # 728,
            "translate": "",  # 726,
            "calendar update": "",  # 726,
            # "change language": "",  # 726,
            "schedule maintenance": "",  # 725,
            "traffic": "",  # 724,
            "flight status": "",  # 724,
            "goodbye": "",  # 723,
            "order checks": "",  # 722,
            "direct deposit": "",  # 722,
            "receiving money": "",  # 721,
            # "change speed": "",  # 721,
            "weather": "",  # 721,
            "book flight": "",  # 721,
            "travel alert": "",  # 720,
            "car rental": "",  # 720,
            # "user name": "",  # 718,
            "compromised card": "",  # 717,
            "order status": "",  # 717,
            # "sync device": "",  # 717,
            "international visa": "",  # 717,
            # "change accent": "",  # 717,
            "vaccines": "",  # 716,
            "tire change": "",  # 715,
            "gas type": "",  # 715,
            "flip coin": "",  # 715,
            # "change user name": "",  # 715,
            "restaurant reservation": "",  # 713,
            "calories": "",  # 712,
            "how old are you": "",  # 711,
            "ingredient substitution": "",  # 711,
            "find phone": "",  # 711,
            "plug type": "",  # 710,
            "calculator": "",  # 709,
            "w2": "",  # 708,
            "smart home": "",  # 707,
            "spelling": "",  # 705,
            "lost luggage": "",  # 705,
            "rollover 401k": "",  # 705,
            "text": "",  # 705,
            "share location": "",  # 705,
            "jump start": "",  # 704,
            "do you have pets": "",  # 704,
            "definition": "",  # 701,
            "uber": "",  # 700,
            "meaning of life": "",  # 697,
            "carry on": "",  # 696,
            "thank you": "",  # 696,
            "mpg": "",  # 696,
            "atm support": "",  # 693,
            "payday": "",  # 693,
            "measurement conversion": "",  # 692,
            "who do you work for": "",  # 691,
            "tire pressure": "",  # 690,
            "what song": "",  # 685,
            "card swallowed": "",  # 571,
            "card acceptance": "",  # 560,
            "virtual card not working": "",  # 490,
            "contactless not working": "",  # 453,
            "atis day name": "",  # 138,
        }  # intent labels --> intent statements (Note: inherent the mapping from existing tasks)

        from tasks.atis import TaskAtis
        from tasks.snips import TaskSnips
        from tasks.banking77 import TaskBanking77
        from tasks.clinc import TaskClinc

        self.task_atis = TaskAtis(
            logger=logger, verbose=verbose, data_dir=data_dir,
            cache_dir=cache_dir, project_root_dir=project_root_dir)
        self.task_snips = TaskSnips(
            logger=logger, verbose=verbose, data_dir=data_dir,
            cache_dir=cache_dir, project_root_dir=project_root_dir)
        self.task_banking77 = TaskBanking77(
            logger=logger, verbose=verbose, data_dir=data_dir,
            cache_dir=cache_dir, project_root_dir=project_root_dir)
        self.task_clinc = TaskClinc(
            logger=logger, verbose=verbose, data_dir=data_dir,
            cache_dir=cache_dir, project_root_dir=project_root_dir)

        for task_obj in [self.task_atis, self.task_snips, self.task_banking77, self.task_clinc]:
            i2s = task_obj.intent_label2statement
            assert isinstance(i2s, dict)
            for k, v in i2s.items():
                if k in self.intent_label2statement:
                    self.intent_label2statement[k] = v

                if task_obj.task_name == "atis":
                    k_new = f"atis {k}"
                    if k_new in self.intent_label2statement:
                        self.intent_label2statement[k_new] = v

        empty_keys = []  # len = 0
        for k, v in self.intent_label2statement.items():
            if v == "":
                empty_keys.append(k)
        self.logger.info(f">>> len(empty_keys) = {len(empty_keys)}\n>>> empty_keys: {empty_keys}")

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {
            # domain: "small talk"
            "who made you", "are you a bot", "what can i ask you",
            # domain: "meta" / "general"
            "change speed", "user name", "whisper mode", "cancel",  # "yes", "no", "maybe"
            "change volume", "change language", "change accent", "change user name", "change ai name",
            "repeat", "sync device", "reset settings",
        }  # Note: adopted from Clinc (while ATIS, SNIPS, and Banking77 do not have intents to skip)

        clinc_domain_filepath = os.path.join(self.raw_data_dir, self.task_name, "clinc_domains.json")
        clinc_domain_info_raw = DataIO.load_json(str(clinc_domain_filepath))
        clinc_domain_info = dict()
        clinc_intent2domain = dict()
        for dk, dv in clinc_domain_info_raw.items():
            assert isinstance(dk, str) and isinstance(dv, list)
            dk_new = str(dk).replace("_", " ").strip()
            dv_new = [str(_v).replace("_", " ").strip() for _v in dv]
            clinc_domain_info[dk_new] = dv_new
            for _v_new in dv_new:
                clinc_intent2domain[_v_new] = dk_new

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            if intent_raw.isupper():
                return [intent_raw.lower()]

            intent_clear = ""
            for ch in intent_raw:
                if ch.isupper():
                    intent_clear += f" {ch.lower()}"
                else:
                    intent_clear += ch

            intent_clear = intent_clear.replace("_", " ").strip()
            intent_clear = intent_clear.replace("-", " ").strip()
            intent_clear = intent_clear.replace("  ", " ").strip()

            return [intent_clear.lower().strip()]

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
                assert file_format == "jsonl"
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                target_split = "dev" if task_split == "valid" else task_split
                cur_data = [_d for _d in cur_data if str(_d["split"]).strip() == target_split]
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item["intent"]).strip()
                cur_intents = json.loads(cur_intent_label.replace("'", "\""))
                assert isinstance(cur_intents, list) and len(cur_intents) > 0
                _cur_intent_labels = [_normalize_intent_label(_i)[0] for _i in cur_intents]
                assert len(_cur_intent_labels) > 0
                _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                if len(_cur_intent_labels) == 0:
                    continue
                for _cur_intent_label in _cur_intent_labels:
                    if _cur_intent_label not in all_intents_dict:
                        all_intents_dict[_cur_intent_label] = 1
                    else:
                        all_intents_dict[_cur_intent_label] += 1

                    cur_text_raw = str(raw_item["utterance"]).strip()
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
                cur_filename = str(cur_filename)
                cur_filepath = os.path.join(self.raw_data_dir, self.task_name, cur_filename)
                file_format = cur_filename.split(".")[-1]
                assert file_format == "jsonl"
                cur_data = DataIO.load_jsonl(str(cur_filepath))
                target_split = "dev" if task_split == "valid" else task_split
                cur_data = [_d for _d in cur_data if str(_d["split"]).strip() == target_split]
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item["utterance"]).strip()

                    cur_intent_label = str(raw_item["intent"]).strip()
                    cur_intents = json.loads(cur_intent_label.replace("'", "\""))
                    assert isinstance(cur_intents, list) and len(cur_intents) > 0
                    _cur_intent_labels = [_normalize_intent_label(_i)[0] for _i in cur_intents]
                    assert len(_cur_intent_labels) > 0
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
                    if "atis" in cur_filename.lower():  # ATIS
                        cur_domain, cur_topic = "daily life", "air travel"
                        cur_domain_topic.append([cur_domain, cur_topic])
                    elif "banking" in cur_filename.lower():  # Banking77
                        cur_domain, cur_topic = "daily life", "banking"
                        cur_domain_topic.append([cur_domain, cur_topic])
                    elif "clinc" in cur_filename.lower():  # CLINC
                        # assert len(_cur_intent_labels) == 1 and _cur_intent_labels[0] in clinc_intent2domain
                        for _cur_intent_label in _cur_intent_labels:
                            assert _cur_intent_label in clinc_intent2domain
                            cur_domain_raw = clinc_intent2domain[_cur_intent_labels[0]]
                            cur_domain = "daily life"
                            cur_topic = ""
                            if cur_domain_raw == "meta":
                                cur_domain = "general"
                                cur_topic = ""
                            if cur_domain_raw == "credit cards":
                                cur_domain = "daily life"
                                cur_topic = "banking"
                            if cur_domain_raw == "auto and commute":
                                cur_domain = "daily life"
                                cur_topic = "navigation"
                            if cur_domain_raw in ["home", "utility", "small talk"]:
                                cur_domain = "daily life"
                                cur_topic = cur_domain_raw
                            cur_domain_topic.append([cur_domain, cur_topic])
                    else:  # SNIPS
                        assert "snips" in cur_filename.lower()
                        for _cur_intent_label in _cur_intent_labels:
                            assert _cur_intent_label in self.task_snips.intent_label2category
                            cur_domain, cur_topic = self.task_snips.intent_label2category[_cur_intent_label]
                            cur_domain_topic.append([cur_domain, cur_topic])

                    for cur_domain, cur_topic in cur_domain_topic:
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
                        "paper_year": 2024,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "query",  # str: query/dialogue/monologue
                        "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
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
