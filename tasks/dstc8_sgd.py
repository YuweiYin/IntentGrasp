# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskDSTC8SGD(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "dstc8_sgd"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "dialogue",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2020,  # The year of publication/preprint
            "paper_url": "https://ojs.aaai.org/index.php/AAAI/article/view/6394",  # URL of the dataset paper
            "license": "CC-BY-SA",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 3: The Schema-Guided Dialogue Dataset
        #   Our simulator framework interacts with these services to generate dialogue outlines, which
        #     are a structured representation of dialogue semantics.
        #   We then used a crowdsourcing procedure to paraphrase these outlines to natural language utterances.
        #   Our novel crowdsourcing procedure preserves all annotations obtained from the simulator and
        #     does not require any extra annotations after dialogue collection.

        self.task_data = {
            "train": {
                "filenames": ["train"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "buy event tickets": "To buy event tickets.",  # 12784,
            "reserve restaurant": "To reserve a restaurant.",  # 12057,
            "book appointment": "To book an appointment.",  # 11697,
            "find events": "To find events.",  # 11463,
            "find bus": "To find the bus.",  # 10711,
            "search hotel": "To search for a hotel.",  # 10199,
            "get ride": "To get a ride.",  # 9976,
            "find provider": "To find the provider.",  # 9934,
            "find restaurants": "To find restaurants.",  # 9873,
            "reserve hotel": "To reserve a hotel.",  # 9675,
            "get cars available": "To get available cars.",  # 9615,
            "find movies": "To find movies.",  # 9441,
            "find attractions": "To find tourist attraction.",  # 9189,
            "search roundtrip flights": "To search for roundtrip flights.",  # 8813,
            "search oneway flight": "To search for oneway flight.",  # 7693,
            "reserve car": "To reserve a car.",  # 7521,
            "buy bus ticket": "To buy bus tickets.",  # 7426,
            "get weather": "To get weather information.",  # 5303,
            "find apartment": "To find an apartment.",  # 4165,
            "get times for movie": "To get times for movie.",  # 4087,
            "play movie": "To play a movie.",  # 3460,
            "schedule visit": "To schedule a visit.",  # 3449,
            "get event dates": "To get event dates.",  # 3406,
            "search house": "To search for a house.",  # 3399,
            "play media": "To play the media.",  # 3321,
            "play song": "To play a song.",  # 3113,
            "reserve roundtrip flights": "To reserve roundtrip flights.",  # 3079,
            "check balance": "To check the balance.",  # 2766,
            "reserve oneway flight": "To reserve an oneway flight.",  # 2446,
            "add event": "To add an event.",  # 2408,
            "book house": "To book a house.",  # 2368,
            "get available time": "To get available time.",  # 2278,
            "lookup music": "To look up the music.",  # 2102,
            "transfer money": "To transfer money.",  # 1744,
            "lookup song": "To look up a song.",  # 1548,
            "find trains": "To find trains.",  # 1476,
            "share location": "To share the location.",  # 1161,
            "get train tickets": "To get train tickets.",  # 764,
            "get events": "To get events information.",  # 700,
            "find home by area": "To find home by area.",  # 663,
            "add alarm": "To add an alarm.",  # 617,
            "make payment": "To make a payment.",  # 606,
            "get alarms": "To get alarms.",  # 583,
            "buy movie tickets": "To buy movie tickets.",  # 473,
            "rent movie": "To rent a movie.",  # 448,
            "request payment": "To request the payment.",  # 438,
        }  # intent labels --> intent statements

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

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

            return [intent_clear.lower().strip()]

        # Load the data (context/query + intent labels)
        # First, get all the unique intents across different splits (train/valid/test)
        all_intents_dict = dict()
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            for cur_dirname in cur_filenames:
                cur_dirpath = os.path.join(self.raw_data_dir, self.task_name, cur_dirname)
                cur_dirpath = str(cur_dirpath)
                assert os.path.isdir(cur_dirpath)
                cur_fn_list = os.listdir(cur_dirpath)
                cur_fn_list.sort()
                cur_fn_list = [_fn for _fn in cur_fn_list if _fn.startswith("dialogues") and _fn.endswith(".json")]
                for cur_filename in cur_fn_list:
                    file_format = cur_filename.split(".")[-1]
                    assert file_format == "json"
                    cur_filepath = os.path.join(cur_dirpath, cur_filename)
                    cur_data = DataIO.load_json(str(cur_filepath))
                    cur_split_raw_data += cur_data  # Note: each item is a multi-turn dialogue

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                for turn in raw_item["turns"]:
                    if not isinstance(turn, dict) or "frames" not in turn:
                        continue
                    if "speaker" not in turn or "utterance" not in turn:
                        continue
                    cur_frames = turn["frames"]
                    for frame in cur_frames:
                        if not isinstance(frame, dict) or "state" not in frame:
                            continue
                        cur_state = frame["state"]
                        if not isinstance(cur_state, dict) or "active_intent" not in cur_state:
                            continue
                        cur_intent_label = str(cur_state["active_intent"]).strip()
                        if cur_intent_label.lower().strip() == "none":
                            continue
                        _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                        for _cur_intent_label in _cur_intent_labels:
                            if _cur_intent_label not in all_intents_dict:
                                all_intents_dict[_cur_intent_label] = 1
                            else:
                                all_intents_dict[_cur_intent_label] += 1

        # Sort the intents dict by the count in descending order
        # all_intents_tuple = list(all_intents_dict.items())
        # all_intents_tuple.sort(key=lambda x: x[1], reverse=True)
        # all_intents_list = [_tuple[0] for _tuple in all_intents_tuple]
        # all_intents_dict = {_k: _v for _k, _v in all_intents_tuple}

        all_domains = dict()
        all_topics = dict()
        all_domain_topic = dict()

        domain_info = dict()
        intent2domain = dict()
        for splits in ["train", "valid", "test"]:
            domain_filepath = os.path.join(self.raw_data_dir, self.task_name, splits, "schema.json")
            domain_info_raw = DataIO.load_json(str(domain_filepath))
            assert isinstance(domain_info_raw, list)
            for domain_dict in domain_info_raw:
                cur_domain_name = str(domain_dict["service_name"]).strip()
                if "_" in cur_domain_name:
                    cur_domain_name = cur_domain_name.split("_")[0].strip()
                cur_domain_name = _normalize_intent_label(cur_domain_name)[0].strip()

                if cur_domain_name not in domain_info:
                    domain_info[cur_domain_name] = []

                cur_domain_intents = list(domain_dict["intents"])
                for _it in cur_domain_intents:
                    cur_intent_name = str(_it["name"]).strip()
                    cur_intent_name = _normalize_intent_label(cur_intent_name)[0].strip()
                    # assert cur_intent_name in all_intents_dict, cur_intent_name
                    assert cur_intent_name in self.intent_label2statement, cur_intent_name

                    domain_info[cur_domain_name].append(cur_intent_name)
                    intent2domain[cur_intent_name] = cur_domain_name

        # assert len(set(all_intents_dict.keys()) - set(intent2domain.keys())) == 0
        # assert len(set(intent2domain.keys()) - set(all_intents_dict.keys())) == 0

        assert len(set(self.intent_label2statement.keys()) - set(intent2domain.keys())) == 0
        assert len(set(intent2domain.keys()) - set(self.intent_label2statement.keys())) == 0

        # Then, obtain the context/query strings and intent labels (with counts) per split
        for task_split in ["train", "valid", "test"]:
            cur_filenames = self.task_data[task_split]["filenames"]
            if not isinstance(cur_filenames, list) or len(cur_filenames) == 0:
                continue
            # cur_split_raw_data = []  # The raw data of the current split (train/valid/test)
            cur_data_processed = []  # The processed data of the current split
            # cur_split_intents = {_it: 0 for _it in all_intents_list}  # The intent counter of the current split
            cur_split_intents = {_it: 0 for _it in self.intent_label2statement.keys()}
            item_idx = 0
            show_cnt = int(1e5)

            for cur_dirname in cur_filenames:
                cur_file_raw_data = []
                cur_dirpath = os.path.join(self.raw_data_dir, self.task_name, cur_dirname)
                cur_dirpath = str(cur_dirpath)
                assert os.path.isdir(cur_dirpath)
                cur_fn_list = os.listdir(cur_dirpath)
                cur_fn_list.sort()
                cur_fn_list = [_fn for _fn in cur_fn_list if _fn.startswith("dialogues") and _fn.endswith(".json")]
                if task_split == "train":
                    # cur_fn_list = cur_fn_list[:30]
                    assert len(cur_fn_list) == 30
                elif task_split == "valid":
                    # cur_fn_list = cur_fn_list[:5]
                    assert len(cur_fn_list) == 5
                elif task_split == "test":
                    # cur_fn_list = cur_fn_list[:5]
                    assert len(cur_fn_list) == 5
                else:
                    raise ValueError(f"Task split {task_split} not supported")
                for cur_filename in cur_fn_list:
                    file_format = cur_filename.split(".")[-1]
                    assert file_format == "json"
                    cur_filepath = os.path.join(cur_dirpath, cur_filename)
                    cur_data = DataIO.load_json(str(cur_filepath))
                    cur_file_raw_data += cur_data  # Note: each item is a multi-turn dialogue

                    # Parse raw data items
                    for raw_item in cur_file_raw_data:
                        dialogue_history = ""

                        for turn_idx, turn in enumerate(raw_item["turns"]):
                            cur_speaker_raw = str(turn["speaker"]).strip()
                            if cur_speaker_raw.lower() == "user":
                                cur_speaker_raw = "user"
                            elif cur_speaker_raw.lower() == "system":
                                cur_speaker_raw = "agent"
                            else:
                                raise ValueError(f"Unknown speaker {cur_speaker_raw}")
                            cur_text_raw = f"{cur_speaker_raw}: " + str(turn["utterance"]).strip()

                            cur_context_raw = dialogue_history
                            dialogue_history = f"{cur_context_raw}\n{cur_text_raw}".strip()

                            if not isinstance(turn, dict) or "frames" not in turn:
                                continue
                            if "speaker" not in turn or "utterance" not in turn:
                                continue

                            cur_frames = turn["frames"]
                            for frame in cur_frames:
                                if not isinstance(frame, dict) or "state" not in frame:
                                    continue
                                cur_state = frame["state"]
                                if not isinstance(cur_state, dict) or "active_intent" not in cur_state:
                                    continue
                                cur_intent_label = str(cur_state["active_intent"]).strip()
                                if cur_intent_label.lower().strip() == "none":
                                    continue
                                _cur_intent_labels = _normalize_intent_label(cur_intent_label)
                                for _cur_intent_label in _cur_intent_labels:
                                    assert _cur_intent_label in cur_split_intents
                                    cur_split_intents[_cur_intent_label] += 1

                                assert len(_cur_intent_labels) == 1 and _cur_intent_labels[0] in intent2domain
                                cur_topic = intent2domain[_cur_intent_labels[0]]

                                if cur_topic == "restaurants":
                                    cur_topic = "restaurant"
                                if cur_topic == "events":
                                    cur_topic = "event"
                                if cur_topic == "movies":
                                    cur_topic = "movie"
                                if cur_topic == "flights":
                                    cur_topic = "flight"
                                if cur_topic == "rental cars":
                                    cur_topic = "rental car"
                                if cur_topic == "buses":
                                    cur_topic = "bus"
                                if cur_topic == "hotels":
                                    cur_topic = "hotel"
                                if cur_topic == "services":
                                    cur_topic = "service"
                                if cur_topic == "homes":
                                    cur_topic = "home"
                                if cur_topic == "banks":
                                    cur_topic = "banking"

                                cur_speaker = cur_speaker_raw
                                assert cur_speaker == "user"
                                iu_context = dialogue_history
                                iu_question_raw = f"What is the intent of the {cur_speaker}?"
                                iu_answer_intent_raw = []
                                for _cur_intent_label in _cur_intent_labels:
                                    assert _cur_intent_label in self.intent_label2statement
                                    iu_answer_intent_raw.append(self.intent_label2statement[_cur_intent_label])

                                cur_domain_topic = []
                                for _cur_intent_label in _cur_intent_labels:
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
                                    "paper_year": 2020,  # The year of publication/preprint
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

        # self.task_meta["all_intents"] = all_intents_dict
        # self.task_meta["num_intents"] = len(all_intents_dict)
        self.task_meta["all_intents"] = self.intent_label2statement
        self.task_meta["num_intents"] = len(self.intent_label2statement)
        self.task_meta["all_domains"] = all_domains
        self.task_meta["all_topics"] = all_topics
        self.task_meta["all_domain_topic"] = all_domain_topic
        self.task_meta["intent_label2statement"] = self.intent_label2statement
        if do_save:
            self.save_processed_data(self.unified_data_dir, verbose=True)
        return None
