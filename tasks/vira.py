# -*- coding: utf-8 -*-

import os
from typing import Optional, List

import random

from tasks import TaskManager
from utils.data_io import DataIO


class TaskVIRA(TaskManager):

    def __init__(
            self,
            logger,
            verbose: bool = False,
            cache_dir: Optional[str] = None,
            project_root_dir: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_root_dir, data_dir)

        self.task_name = "vira"
        self.task_meta = {
            "name": self.task_name,
            "text_form": "query",  # str: query/dialogue/monologue
            "intent_type": "single",  # "multiple" if multiple intents per item else "single"
            "is_synthetic": False,  # True if the dataset is synthetic (not human annotated)
            "is_sensitive": False,  # True if the dataset contains sensitive/harmful text
            "paper_year": 2023,  # The year of publication/preprint
            "paper_url": "https://aclanthology.org/2023.findings-eacl.100/",  # URL of the dataset paper
            # "The intent classifier was trained on data collected from crowd annotators using the Appen platform."
            "license": "Apache",  # the releasing license of the original dataset
            "intent_description": {},  # The description of each intent label
        }
        # Section 6.1: The Oracle
        #   For the Oracle we use VIRA’s intent classifier.
        #   For each intent amongst the final 181 intents covered by VIRA, we asked 18 Appen crowd annotators to
        #     contribute three different intent expressions,
        # Section 7.1: The Oracle
        #   We asked 3 annotators to annotate whether a given pair of texts has a similar intent or meaning,
        #     and took the majority vote as the ground-truth.
        #   The accuracy of the Oracle on this data is 0.85.
        #   We use the same annotation task as in (i). The accuracy of the Oracle on this data is 0.86.
        # Appendix F
        #   In addition, we included test questions of text pairs manually selected from the training data
        #     of the Oracle, and annotators with less than 70% accuracy on them were removed from the task.

        self.task_data = {
            "train": {
                "filenames": ["train.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),  # key: the normalized intent label; value: the count of this intent label.
            },
            "valid": {
                "filenames": ["valid.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            },
            "test": {
                "filenames": ["test.parquet"],
                "data": [],
                "num_data": 0,
                "intents": dict(),
            }
        }

        self.max_num_test = int(1e4)

        intent_label2statement_raw = {
            "COVID-19 is not as dangerous as they say": "To express:",
            "Do I need to continue safety measures after getting the vaccine?": "To ask:",
            "How long until I will be protected after taking the vaccine?": "To ask:",
            "How many people already got the vaccine?": "To ask:",
            "I am afraid the vaccine will change my DNA": "To express:",
            "I am concerned getting the vaccine because I have a pre-existing condition": "To express:",
            "I am concerned I will be a guinea pig": "To express:",
            "I'm concerned the vaccine will make me sick.": "To express:",
            "I am not sure if I can trust the government": "To express:",
            "I am young and healthy so I don't think I should vaccinate": "To express:",
            "I distrust this vaccine": "To express:",
            "How much will I have to pay for the vaccine": "To ask:",
            "I don't think the vaccine is necessary": "To express:",
            "I don't trust the companies producing the vaccines": "To express:",
            "I don't want my children to get the vaccine": "To express:",
            "I think the vaccine was not tested on my community": "To express:",
            "I'm not sure the vaccine is effective enough": "To express:",
            "I'm waiting to see how it affects others": "To express:",
            "COVID vaccines can be worse than the disease itself": "To express:",
            "Long term side-effects were not researched enough": "To express:",
            "Are regular safety measures enough to stay healthy?": "To ask:",
            "Should people that had COVID get the vaccine?": "To ask:",
            "Side effects and adverse reactions worry me": "To express:",
            "The COVID vaccine is not safe": "To express:",
            "The vaccine should not be mandatory": "To express:",
            "Do vaccines work against the mutated strains of COVID-19?": "To ask:",
            "They will put a chip/microchip to manipulate me": "To express:",
            "What can this chatbot do?": "To ask:",
            "What is in the vaccine?": "To ask:",
            "Which one of the vaccines should I take?": "To ask:",
            "Will I test positive after getting the vaccine?": "To ask:",
            "Can other vaccines protect me from COVID-19?": "To ask:",
            "Do I qualify for the vaccine?": "To ask:",
            "I don't trust vaccines if they're from China or Russia": "To express:",
            "Are the side effects worse for the second shot": "To ask:",
            "Can I get a second dose even after a COVID exposure?": "To ask:",
            "Can I get other vaccines at the same time?": "To ask:",
            "Can I get the vaccine if I have allergies?": "To ask:",
            "Can I get the vaccine if I have had allergic reactions to vaccines before?": "To ask:",
            "Can I have the vaccine as a Catholic?": "To ask:",
            "Can I have the vaccine if I'm allergic to penicillin?": "To ask:",
            "Can I still get COVID even after being vaccinated?": "To ask:",
            "Can you mix the vaccines?": "To ask:",
            "COVID-19 vaccines cause brain inflammation": "To express:",
            "Do the COVID-19 vaccines cause Bell's palsy?": "To ask:",
            "Do the mRNA vaccines contain preservatives, like thimerosal?": "To ask:",
            "Do the vaccines work in obese people?": "To ask:",
            "Do you have to be tested for COVID before you vaccinated?": "To ask:",
            "Does the vaccine contain animal products?": "To ask:",
            "Does the vaccine contain live COVID virus?": "To ask:",
            "Does the vaccine impact pregnancy?": "To ask:",
            "Does the vaccine work if I do not experience any side effects?": "To ask:",
            "How can I stay safe until I'm vaccinated?": "To ask:",
            "How do I know I'm getting a legitimate, authorized vaccine?": "To ask:",
            "How do I report an adverse reaction or side-effect": "To ask:",
            "How long do I have to wait between doses?": "To ask:",
            "How many doses do I need?": "To ask:",
            "How was the vaccine tested?": "To ask:",
            "I am concerned about getting the vaccine because of my medications.": "To express:",
            "I don't want the v-safe app monitoring or tracking me": "To express:",
            "I don't want to share my personal information": "To express:",
            "Is breastfeeding safe with the vaccine": "To ask:",
            "Is the Johnson & Johnson vaccine less effective than the others?": "To ask:",
            "Is the vaccine halal?": "To ask:",
            "Is the vaccine Kosher?": "To ask:",
            "Is there vaccine safety monitoring?": "To ask:",
            "Other vaccines have caused long-term health problems": "To express:",
            "Should I get the COVID-19 vaccine if I am immunocompromised": "To ask:",
            "Should I get the vaccine if I've tested positive for antibodies?": "To ask:",
            "The vaccine includes fetal tissue or abortion by-products": "To express:",
            "The vaccine was rushed": "To express:",
            "Vaccine side effects are not getting reported": "To express:",
            "What does vaccine efficacy mean?": "To ask:",
            "What if I still get infected even after receiving the vaccine?": "To ask:",
            "What if I've been treated with convalescent plasma?": "To ask:",
            "What if I've been treated with monoclonal antibodies?": "To ask:",
            "What is mRNA?": "To ask:",
            "What is the difference between mRNA and viral vector vaccines?": "To ask:",
            "When can I go back to normal life?": "To ask:",
            "Why are there different vaccines?": "To ask:",
            "Why do I need the COVID vaccine if I don't get immunized for flu": "To ask:",
            "Why do we need the vaccine if we can wait for herd immunity?": "To ask:",
            "Why get vaccinated if I can still transmit the virus?": "To ask:",
            "Will 1 dose of vaccine protect me?": "To ask:",
            "Can I take a pain reliever when I get vaccinated?": "To ask:",
            "Will the vaccine benefit me?": "To ask:",
            "Will the vaccine make me sterile or infertile?": "To ask:",
            "Can we change the vaccine quickly if the virus mutates?": "To ask:",
            "Can I get COVID-19 from the vaccine?": "To ask:",
            "I'm still experiencing COVID symptoms even after testing negative - should I still take the vaccine?": "To ask:",
            "Can children get the vaccine?": "To ask:",
            "Can we choose which vaccine we want?": "To ask:",
            "How long does the immunity from the vaccine last?": "To ask:",
            "The mortality rate of COVID-19 is low, why should I get the vaccine?": "To ask:",
            "There are many reports of severe side effects or deaths from the vaccine": "To express:",
            "How can I get the vaccine?": "To ask:",
            "I am worried about blood clots as a result of the vaccine": "To express:",
            "what is covid?": "To ask:",
            "Who developed the vaccine?": "To ask:",
            "Which vaccines are available?": "To ask:",
            "What are the side effect of the vaccine?": "To ask:",
            "Can I meet in groups after I'm vaccinated?": "To ask:",
            "Is it safe to go to the gym indoors if I'm vaccinated?": "To ask:",
            "How do I protect myself indoors?": "To ask:",
            "What are the effects of long COVID?": "To ask:",
            "Do you need a social security number to get a COVID-19 vaccine?": "To ask:",
            "Do you need to be a U.S. citizen to get a COVID-19 vaccine?": "To ask:",
            "Is it okay for me to travel internationally if I'm vaccinated?": "To ask:",
            "Can my kids go back to school without a vaccine?": "To ask:",
            "Will I need a booster shot?": "To ask:",
            "If I live with an immuno-compromised individual, do I still need to wear a mask outdoors if I'm vaccinated?": "To ask:",
            "Does the vaccine prevent transmission?": "To ask:",
            "Why is AstraZeneca not approved in the USA?": "To ask:",
            "Do I need to change my masking and social distancing practices depending on which COVID-19 vaccine I got?": "To ask:",
            "Does the Pfizer vaccine cause myocarditis?": "To ask:",
            "Does the Pfizer vaccine cause heart problems?": "To ask:",
            "What can you tell me about COVID-19 vaccines?": "To ask:",
            "Are there medical contraindications to the vaccines?": "To ask:",
            "How many people died from COVID-19?": "To ask:",
            "What about reports of abnormal periods due to the vaccine?": "To ask:",
            "Do I need the vaccine?": "To ask:",
            "Tell me about the vaccine": "To ask:",
            "Is the Pfizer vaccine safe for young men?": "To ask:",
            "Will vaccination lead to more dangerous variants?": "To ask:",
            "Is it safe for my baby to get the vaccine?": "To ask:",
            "Did a volunteer in the Oxford trial die?": "To ask:",
            "Can I get COVID-19 twice?": "To ask:",
            "Are some vaccines safer for younger children than others?": "To ask:",
            "How long am I immune from COVID-19 if I had the virus?": "To ask:",
            "Are women more likely to get worse side effects than men?": "To ask:",
            "How do I convince my family and friends to get the COVID-19 vaccine?": "To ask:",
            "Why are COVID-19 vaccination rates slowing in the U.S.?": "To ask:",
            "I'm going to get vaccinated": "To express:",
            "Is getting vaccinated painful?": "To ask:",
            "What do I do if I lose my COVID-19 vaccination card?": "To ask:",
            "Can I get swollen lymph nodes from the vaccine?": "To ask:",
            "Can my newborn become immune to COVID-19 if I'm vaccinated?": "To ask:",
            "COVID-19 is over, why should I get the vaccine?": "To ask:",
            "Did one woman die after getting the J&J vaccine?": "To ask:",
            "Do people become magnetic after getting vaccinated?": "To ask:",
            "Does the vaccine contain eggs?": "To ask:",
            "How is the COVID-19 vaccine different than others?": "To ask:",
            "How soon after I've had COVID-19 can I get the vaccination?": "To ask:",
            "Is it safe for my teen to get the vaccine?": "To ask:",
            "Is this Pfizer vaccine equally effective in kids as it is in adults?": "To ask:",
            "Were the COVID-19 vaccines tested on animals?": "To ask:",
            "What are the side effects of the vaccine in children?": "To ask:",
            "What is the delta variant?": "To ask:",
            "What is the J&J vaccine?": "To ask:",
            "What is the Moderna vaccine?": "To ask:",
            "What is the Pfizer vaccine?": "To ask:",
            "Where are we required to wear masks now?": "To ask:",
            "Who can get the Pfizer vaccine?": "To ask:",
            "Who can I talk to about COVID-19 in person?": "To ask:",
            "Why should I trust you?": "To ask:",
            "Will my child need my permission to get vaccinated?": "To ask:",
            "Will the US reach herd immunity?": "To ask:",
            "Will my child miss school when they get vaccinated?": "To ask:",
            "Is the vaccine FDA approved?": "To ask:",
            "Why do vaccinated people need to wear a mask indoors?": "To ask:",
            "Do vaccinated people need to quarantine if exposed to COVID-19?": "To ask:",
            "What is Ivermectin?": "To ask:",
            "Does the Johnson and Johnson vaccine cause Rare Nerve Syndrome?": "To ask:",
            "What is the difference between quarantine and isolation?": "To ask:",
            "Does the COVID-19 vaccine cause autism?": "To ask:",
            "Does the vaccine cause impotence?": "To ask:",
            "Who is required to get vaccinated under the federal vaccine mandate?": "To ask:",
            "Is the Delta variant more dangerous for kids?": "To ask:",
            "Will there be a booster shot for J&J and Moderna?": "To ask:",
            "Is the booster the same as the original vaccine?": "To ask:",
            "What are the side effects of booster shots?": "To ask:",
            "What is the difference between the third shot and a booster shot?": "To ask:",
            "How common are vaccine side effects?": "To ask:",
            "Why do my kids need a vaccine if they're unlikely to get sick with COVID-19?": "To ask:",
            "What happens if there is a COVID-19 case at my child's school?": "To ask:",
            "Are booster shot side effects worse than those from the second shot?": "To ask:",
            "Is the booster shot dangerous?": "To ask:",
            "Can I get the vaccine if I have Multiple Sclerosis?": "To ask:",
            "Do children receive the same dose of Pfizer as adults?": "To ask:",
            "What is the Omicron variant?": "To ask:",
            "How effective is the vaccine against the Omicron variant?": "To ask:",
        }   # intent labels --> intent statements

        self.intent_label2statement = dict()
        for k, v in intent_label2statement_raw.items():
            self.intent_label2statement[k] = f"{v.strip()} {k.strip()}"

    def raw_data_unification(
            self,
            do_save: bool = False,
    ) -> None:
        self.logger.info(f">>> [task_name: {self.task_name}]")
        assert isinstance(self.task_meta, dict) and isinstance(self.task_data, dict)

        skip_intents = {}

        intent_id2name = dict()
        intent_name2id = dict()
        intents_info_filepath = os.path.join(self.raw_data_dir, self.task_name, "intents.csv")
        intents_info_file = DataIO.load_csv(str(intents_info_filepath), delimiter=",")
        # csv_header = intents_info_file[0]
        intents_info_file = intents_info_file[1:]  # ignore the csv header
        for i_id, i_names in enumerate(intents_info_file):
            assert isinstance(i_names, list) and len(i_names) == 1
            i_name = str(i_names[0]).strip()
            if i_name in skip_intents:
                continue
            intent_id2name[i_id] = i_name
            intent_name2id[i_name] = i_id

        def _normalize_intent_label(intent_raw: str) -> List[str]:
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
                assert file_format == "parquet"
                cur_data = DataIO.load_parquet(str(cur_filepath))
                cur_data_list = []
                for _t, _i in zip(cur_data["text"].tolist(), cur_data["label"].tolist()):
                    _i = int(_i)
                    assert _i in intent_id2name
                    _i_name = str(intent_id2name[_i]).strip()
                    if _i_name in skip_intents:
                        continue
                    cur_data_list.append({
                        "text": str(_t).strip(),
                        "intent": _i_name,
                    })
                cur_split_raw_data += cur_data_list

            # Record intent labels
            for idx, raw_item in enumerate(cur_split_raw_data):
                cur_intent_label = str(raw_item["intent"]).strip()
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

        assert len(set(all_intents_dict.keys()) - set(intent_name2id.keys())) == 0
        assert len(set(intent_name2id.keys()) - set(all_intents_dict.keys())) == 0

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
                assert file_format == "parquet"
                cur_data = DataIO.load_parquet(str(cur_filepath))
                cur_data_list = []
                for _t, _i in zip(cur_data["text"].tolist(), cur_data["label"].tolist()):
                    _i = int(_i)
                    assert _i in intent_id2name
                    _i_name = str(intent_id2name[_i]).strip()
                    if _i_name in skip_intents:
                        continue
                    cur_data_list.append({
                        "text": str(_t).strip(),
                        "intent": _i_name,
                    })
                cur_file_raw_data = cur_data_list

                # Parse raw data items
                for raw_item in cur_file_raw_data:
                    cur_text_raw = str(raw_item["text"]).strip()
                    cur_intent_label = str(raw_item["intent"]).strip()

                    _cur_intent_labels = _normalize_intent_label(cur_intent_label)
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
                    cur_domain, cur_topic = "coronavirus pandemic", "COVID-19 vaccine"  # "COVID-19 vaccine", ""
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
