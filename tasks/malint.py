# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskMalInt(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "malint"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "monologue",  # str: query/dialogue/monologue
            "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2026,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2026.eacl-long.144/",  # URL of the dataset paper
            "license": "CC-BY",  # the releasing license of the original dataset (ACL publication - CC-BY-NC-SA)
            "intent_description": {
                "UCPI": "Undermining the credibility of public institutions [UCPI] - "
                        "This can be done by undermining official communications, insinuating bad intentions or "
                        "falsely exposing corruption. The idea is to make citizens disbelieve in the effectiveness "
                        "of their own state, undermine the sense of its existence or actively fight against it. "
                        "This is ultimately meant to lead to resentment of the system, thus undermining the very "
                        "essence of democracy. As a result, it becomes easier to spread false information, and "
                        "the public's resistance to outside influence decreases.",
                "CPV": "Changing political views [CPV] - "
                       "Influencing voter preferences is a common procedure used by authors. "
                       "Changing political beliefs is aimed at strengthening one side of a political dispute "
                       "and arousing resentment against others. It usually involves the simultaneous promotion "
                       "of politicians from extremist movements, which are treated as an alternative to the major "
                       "parties. It is often based on the portrayal of mainstream politicians as corrupt and evil "
                       "to the bone (e.g., portraying them as traitors to the nation, dependent on the outside "
                       "influence of global elites).",
                "UIOA": "Undermining international organizations and alliances [UIOA] - "
                        "Undermining the credibility of international institutions is often part of activities "
                        "carried out by external forces. These are aimed at breaking up alliances of democratic "
                        "states to facilitate propaganda efforts by authoritarian states. Numerous extreme "
                        "political movements also have an interest in shattering trust in international institutions. "
                        "This is part of a populist influence on society and a way to gain power. International "
                        "institutions are then most often portrayed as entities that take away the sovereignty "
                        "of member states.",
                "PSSA": "Promoting social stereotypes/antagonisms [PSSA] - "
                        "Deepening social divisions is a frequent goal of malicious actors' efforts. "
                        "A strongly divided society is less resistant to manipulation, and mutual distrust "
                        "also promotes a collapse of confidence in the institution of the state and democracy. "
                        "This causes internal problems to absorb most of the attention, giving room for external "
                        "centers of influence to operate. This can take the form of reinforcing xenophobia. "
                        "Aversion to specific social groups can also be exploited.",
                "PASV": "Promoting anti-scientific views [PASV] - "
                        "Science is a frequent enemy of malicious actors. Science enhances critical thinking and "
                        "is an important part of the strength of democracies. Presenting it as an enemy aids in "
                        "undermining the system under which democracies operate. Reinforcing anti-scientific "
                        "attitudes also enables short-term financial gain. The fight against science can be based on "
                        "a direct attack on scientists, but is also a significant element of conspiracy theories.",
                "Credible": "Credible information; not malicious.",
            },  # The description of each intent label
        }
        # Section 2.3: Annotation and Data Quality Control
        #   To ensure annotation reliability and reduce bias, each article was independently reviewed by
        #     two annotators (a primary annotator and their supervisor).
        #   The supervisor performed a third pass, considering independent annotations.
        #   In the event of disagreement, supervisors were encouraged to consult with the initial annotators and,
        #     as needed, with a senior fact-checking expert.
        #   In the first stage, two annotators achieved an agreement of approximately 85.31% on the credibility task.
        #     They reached 65.19% agreement on the more complex multilabel intent task.
        #   In the second stage, the supervisor performed a third annotation. Disagreements were resolved through
        #     consensus, and expert input was utilized when necessary. At this stage, the annotation process
        #     concluded with agreement exceeding 95% between supervisors for both tasks.
        #     This improved the reliability and quality of the dataset.

        self.task_data = {
            "train": {
                "filenames": ["train.csv"],
                # "filenames": [],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.csv"],
                # "filenames": [],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.csv"],
                # "filenames": ["train.csv", "valid.csv", "test.csv"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        self.intent_label2statement = {
            "Credible": "Credible information: not disinformation, no malicious intent.",  # 1016,
            "UCPI": "To undermine the credibility of public institutions.",  # 321,
            "UIOA": "To undermine international organizations and alliances.",  # 234,
            "PSSA": "To promote social stereotypes or antagonisms.",  # 222,
            "CPV": "To change political views.",  # 197,
            "PASV": "To promote anti-scientific views.",  # 154,
        }  # intent labels --> intent statements

        self.intent_label2category = {
            "Credible": ["news", "disinformation"],
            "UCPI": ["news", "disinformation"],
            "UIOA": ["news", "disinformation"],
            "PSSA": ["news", "disinformation"],
            "CPV": ["news", "disinformation"],
            "PASV": ["news", "disinformation"],
        }  # intent labels --> intent categories (domains & topics)

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        def _normalize_intent_label(intent_raw: List[str]) -> List[str]:
            res_intents = [_intent.strip() for _intent in intent_raw]
            return res_intents

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
                # ["article_id", "prior_2024", "article_type", "article_publication_date", "label",
                # "article_topic", "article_title", "article_url", "intention_type", "article_body",
                # "content", "CPV", "PSSA", "UIOA", "PASV", "UCPI"]
                cur_data = cur_data[1:]  # ignore the csv header
                cur_split_raw_data += cur_data

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_disinfo_label = str(raw_item[4]).strip()
                # _cur_intent_labels = list(raw_item[8])
                _cur_intent_labels = []
                if int(raw_item[11]) == 1:
                    _cur_intent_labels.append("CPV")
                if int(raw_item[12]) == 1:
                    _cur_intent_labels.append("PSSA")
                if int(raw_item[13]) == 1:
                    _cur_intent_labels.append("UIOA")
                if int(raw_item[14]) == 1:
                    _cur_intent_labels.append("PASV")
                if int(raw_item[15]) == 1:
                    _cur_intent_labels.append("UCPI")

                if len(_cur_intent_labels) == 0:
                    # assert cur_disinfo_label == "Credible information"
                    if cur_disinfo_label != "Credible information":
                        # Bad data point: Disinformation without intent labels
                        continue
                    _cur_intent_labels.append("Credible")
                else:
                    assert cur_disinfo_label == "Disinformation"

                for _cur_intent_label in _normalize_intent_label(_cur_intent_labels):
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
                # ["article_id", "prior_2024", "article_type", "article_publication_date", "label",
                # "article_topic", "article_title", "article_url", "intention_type", "article_body",
                # "content", "CPV", "PSSA", "UIOA", "PASV", "UCPI"]
                cur_data = cur_data[1:]  # ignore the csv header
                cur_file_raw_data = cur_data

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item[10]).replace("\n", " ").strip()
                    cur_disinfo_label = str(raw_item[4]).strip()
                    _cur_intent_labels = []
                    if int(raw_item[11]) == 1:
                        _cur_intent_labels.append("CPV")
                    if int(raw_item[12]) == 1:
                        _cur_intent_labels.append("PSSA")
                    if int(raw_item[13]) == 1:
                        _cur_intent_labels.append("UIOA")
                    if int(raw_item[14]) == 1:
                        _cur_intent_labels.append("PASV")
                    if int(raw_item[15]) == 1:
                        _cur_intent_labels.append("UCPI")

                    if len(_cur_intent_labels) == 0:
                        # assert cur_disinfo_label == "Credible information"
                        if cur_disinfo_label != "Credible information":
                            # Bad data point: Disinformation without intent labels
                            continue
                        _cur_intent_labels.append("Credible")
                    else:
                        assert cur_disinfo_label == "Disinformation"

                    for _cur_intent_label in _normalize_intent_label(_cur_intent_labels):
                        assert _cur_intent_label in cur_split_intents
                        cur_split_intents[_cur_intent_label] += 1

                    cur_speaker = "news"  # reporter
                    iu_context = f"### News:\n{cur_text_raw}"
                    iu_question_raw = (f"Does the news provide credible information or disinformation? "
                                       f"If it contains disinformation, what is the intent of providing it?")
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
                        "paper_year": 2026,  # The year of publication/preprint
                        "original_task": self.task_name,  # str
                        "original_split": task_split,  # str
                        "text_form": "monologue",  # str: query/dialogue/monologue
                        "intent_type": "multiple",  # "multiple" if multiple intents per item else "single"
                        "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
                        "is_sensitive": True,  # True if the dataset contains sensitive/harmful text
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
