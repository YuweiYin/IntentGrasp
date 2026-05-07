# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskClinc(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "clinc"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2019,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/D19-1131/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 2: Dataset
        #   We defined the intents with guidance from queries collected using a scoping crowdsourcing task,
        #     which prompted crowd workers to provide questions and commands related to topic domains
        #     in the manner they would interact with an artificially intelligent assistant.

        self.task_data = {
            "train": {
                "filenames": ["data_full.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["data_full.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["data_full.json"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            # domain: "banking"
            "freeze account": "To freeze the account.",  # 150
            "routing": "To inquire about routing.",  # 150
            "pin change": "To change the pin or inquire about pin change.",  # 150
            "bill due": "To check the bill due.",  # 150
            "pay bill": "To pay the bill.",  # 150
            "account blocked": "To block an account or inquire about the blocked account.",  # 150
            "interest rate": "To inquire about the interest rate.",  # 150
            "min payment": "To inquire about the minimum payment.",  # 150
            "bill balance": "To inquire about the bill balance.",  # 150
            "transfer": "To transfer money.",  # 150
            "order checks": "To order checks.",  # 150
            "balance": "To inquire about the balance.",  # 150
            "spending history": "To inquire about the spending history.",  # 150
            "transactions": "To inquire about or make transactions.",  # 150
            "report fraud": "To report fraud.",  # 150

            # domain: "credit cards"
            "replacement card duration": "To inquire about the replacement card duration.",  # 150
            "expiration date": "To inquire about the expiration date.",  # 150
            "damaged card": "To inquire about the damaged card.",  # 150
            "improve credit score": "To improve the credit score.",  # 150
            "report lost card": "To report the lost card.",  # 150
            "card declined": "To inquire about the decline of the card.",  # 150
            "credit limit change": "To change the credit limit or inquire about the limit change.",  # 150
            "apr": "To inquire about the annual percentage rate (APR).",  # 150
            "redeem rewards": "To redeem rewards.",  # 150
            "credit limit": "To inquire about the credit limit.",  # 150
            "rewards balance": "To inquire about the rewards balance.",  # 150
            "application status": "To check the application status.",  # 150
            "credit score": "To inquire about the credit score.",  # 150
            "new card": "To open a new card or inquire about related issues.",  # 150
            "international fees": "To inquire about the international fees.",  # 150

            # domain: "kitchen and dining"
            "food last": "To ask how long the food lasts.",  # 150
            "confirm reservation": "To confirm the reservation.",  # 150
            "how busy": "To ask how busy the place is.",  # 150
            "ingredients list": "To ask for the ingredients list.",  # 150
            "calories": "To ask about the calories.",  # 150
            "nutrition info": "To ask about the nutrition information.",  # 150
            "recipe": "To ask for the recipe.",  # 150
            "restaurant reviews": "To check the restaurant reviews.",  # 150
            "restaurant reservation": "To make the restaurant reservation.",  # 150
            "meal suggestion": "To ask for a meal suggestion.",  # 150
            "restaurant suggestion": "To ask for a restaurant suggestion.",  # 150
            "cancel reservation": "To cancel the reservation.",  # 150
            "ingredient substitution": "To require substituting the ingredient.",  # 150
            "cook time": "To ask for the cook time.",  # 150
            "accept reservations": "To accept the reservations.",  # 150

            # domain: "home"
            "what song": "To ask what the song is.",  # 150
            "play music": "To play music.",  # 150
            "todo list update": "To update the todo list.",  # 150
            "reminder": "To set the reminder.",  # 150
            "reminder update": "To update the reminder.",  # 150
            "calendar update": "To update the calendar.",  # 150
            "order status": "To check the order status.",  # 150
            "update playlist": "To update the playlist.",  # 150
            "shopping list": "To make or check the shopping list.",  # 150
            "calendar": "To check the schedule on the calendar.",  # 150
            "next song": "To play the next song.",  # 150
            "order": "To make an order.",  # 150
            "todo list": "To make or check a todo list.",  # 150
            "shopping list update": "To update the shopping list.",  # 150
            "smart home": "To command the smart home assistant.",  # 150

            # domain: "auto and commute"
            "current location": "To check the current location.",  # 150
            "oil change when": "To ask when to change oil.",  # 150
            "oil change how": "To ask how to change oil.",  # 150
            "uber": "To book an Uber.",  # 150
            "traffic": "To ask about traffic information.",  # 150
            "tire pressure": "To check the tire pressure.",  # 150
            "schedule maintenance": "To schedule a maintenance.",  # 150
            "gas": "To check the gas status.",  # 150
            "mpg": "To ask about the mileage per gallon (MPG) of gas.",  # 150
            "distance": "To ask about the distance.",  # 150
            "directions": "To ask about the directions.",  # 150
            "last maintenance": "To check information about the last maintenance.",  # 150
            "gas type": "To ask about the gas type.",  # 150
            "tire change": "To ask when to change the tire.",  # 150
            "jump start": "To ask about jump-starting a battery.",  # 150

            # domain: "travel"
            "plug type": "To ask about the plug type.",  # 150
            "travel notification": "To notify of the travel plan.",  # 150
            "translate": "To translate.",  # 150
            "flight status": "To check the flight status.",  # 150
            "international visa": "To check information about international visa",  # 150
            "timezone": "To check the timezone.",  # 150
            "exchange rate": "To ask about the exchange rate.",  # 150
            "travel suggestion": "To ask about travel suggestions.",  # 150
            "travel alert": "To ask about any alerts for travel.",  # 150
            "vaccines": "To check information about vaccines.",  # 150
            "lost luggage": "To check information about lost luggage.",  # 150
            "book flight": "To book a flight.",  # 150
            "book hotel": "To book a hotel.",  # 150
            "carry on": "To check information about carry-on.",  # 150
            "car rental": "To check information about car rental.",  # 150

            # domain: "utility"
            "weather": "To check the weather.",  # 150
            "alarm": "To set the alarm.",  # 150
            "date": "To check the date.",  # 150
            "find phone": "To find the phone.",  # 150
            "share location": "To share the location.",  # 150
            "timer": "To set a timer.",  # 150
            "make call": "To make a call.",  # 150
            "calculator": "To calculate.",  # 150
            "definition": "To find out the definition or meaning of something.",  # 150
            "measurement conversion": "To convert the measurement.",  # 150
            "flip coin": "To flip a coin.",  # 150
            "spelling": "To spell something or ask about the spelling.",  # 150
            "time": "To check the time.",  # 150
            "roll dice": "To roll the dice.",  # 150
            "text": "To send a text.",  # 150

            # domain: "work"
            "pto request status": "To check the status of the paid time off (PTO) request.",  # 150
            "next holiday": "To check the next holiday.",  # 150
            "insurance change": "To change the insurance.",  # 150
            "insurance": "To ask information about the insurance.",  # 150
            "meeting schedule": "To check the meeting schedule.",  # 150
            "payday": "To check the payday.",  # 150
            "taxes": "To check the taxes.",  # 150
            "income": "To ask about the income.",  # 150
            "rollover 401k": "To rollover or transfer 401(k).",  # 150
            "pto balance": "To check the paid time off (PTO) balance.",  # 150
            "pto request": "To request paid time off (PTO).",  # 150
            "w2": "To ask about the W-2 form.",  # 150
            "schedule meeting": "To schedule a meeting.",  # 150
            "direct deposit": "To ask about the direct deposit.",  # 150
            "pto used": "To check the used paid time off (PTO).",  # 150

            # domain: "small talk"
            # "who made you": "To ask who made you.",  # 150
            "meaning of life": "To talk about the meaning of life.",  # 150
            "who do you work for": "To ask who you work for.",  # 150
            "do you have pets": "To talk about pets.",  # 150
            "what are your hobbies": "To talk about hobbies.",  # 150
            "fun fact": "To talk about fun facts.",  # 150
            "what is your name": "To ask your name.",  # 150
            "where are you from": "To ask where you come from.",  # 150
            "goodbye": "To say goodbye.",  # 150
            "thank you": "To express gratitude.",  # 150
            "greeting": "To express greetings.",  # 150
            "tell joke": "To tell jokes.",  # 150
            # "are you a bot": "To check if you are a bot.",  # 150
            "how old are you": "To ask about the age.",  # 150
            # "what can i ask you": "To check what I can ask you.",  # 150
        }  # intent labels --> intent statements

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
            "change speed", "user name", "whisper mode", "yes", "no", "maybe", "cancel",
            "change volume", "change language", "change accent", "change user name", "change ai name",
            "repeat", "sync device", "reset settings",
        }

        domain_filepath = os.path.join(self.raw_data_dir, self.task_name, "domains.json")
        domain_info_raw = DataIO.load_json(str(domain_filepath))
        domain_info = dict()
        intent2domain = dict()
        for dk, dv in domain_info_raw.items():
            assert isinstance(dk, str) and isinstance(dv, list)
            dk_new = str(dk).replace("_", " ").strip()
            dv_new = [str(_v).replace("_", " ").strip() for _v in dv]
            domain_info[dk_new] = dv_new
            for _v_new in dv_new:
                intent2domain[_v_new] = dk_new

        def _normalize_intent_label(intent_raw: str) -> List[str]:
            intent_raw = intent_raw.strip()
            intent_raw = intent_raw.replace("_", " ")
            return [intent_raw.strip()]

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
                if task_split == "train":
                    cur_split_raw_data += cur_data["train"]
                elif task_split == "valid":
                    cur_split_raw_data += cur_data["val"]
                elif task_split == "test":
                    cur_split_raw_data += cur_data["test"]
                else:
                    raise ValueError(f"Unknown task split: {task_split}")

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item[1]).strip()
                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                if len(_cur_intent_labels) == 0:
                    continue
                for _cur_intent_label in _cur_intent_labels:
                    assert _cur_intent_label in intent2domain
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
                assert file_format == "json"
                cur_data = DataIO.load_json(str(cur_filepath))
                if task_split == "train":
                    cur_file_raw_data = cur_data["train"]
                elif task_split == "valid":
                    cur_file_raw_data = cur_data["val"]
                elif task_split == "test":
                    cur_file_raw_data = cur_data["test"]
                else:
                    raise ValueError(f"Unknown task split: {task_split}")

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item[0]).strip()
                    cur_intent_label = str(raw_item[1]).strip()
                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                    _cur_intent_labels = [_item for _item in _cur_intent_labels if _item not in skip_intents]
                    if len(_cur_intent_labels) == 0:
                        continue
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    assert len(_cur_intent_labels) == 1 and _cur_intent_labels[0] in intent2domain
                    cur_domain_raw = intent2domain[_cur_intent_labels[0]]
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

                    cur_speaker = "user"
                    iu_context = f"{cur_speaker}: {cur_text_raw}"
                    iu_question_raw = f"What is the intent of the {cur_speaker}?"
                    iu_answer_intent_raw = []
                    for _cur_intent_label in _cur_intent_labels:
                        assert _cur_intent_label in self.intent_label2statement
                        iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

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
                        "paper_year": 2019,  # The year of publication/preprint
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
