#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import fire
import json
from typing import Optional, Union
import pandas as pd


def load_json(
        filepath: str,
        mode: str = "r",
        encoding: str = "utf-8",
        verbose: bool = False,
        errors: Optional[str] = None,
) -> Union[list, dict]:
    results = []
    if os.path.isfile(filepath):
        if verbose:
            print(f">>> [load_json] {filepath}")
        with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
            results = json.load(fp_in)
    else:
        if verbose:
            print(f">>> [load_json] filepath does not exist: {filepath}")
    return results


def load_jsonl(
        filepath: str,
        mode: str = "r",
        encoding: str = "utf-8",
        verbose: bool = False,
        errors: Optional[str] = None,
) -> list:
    results = []
    if os.path.isfile(filepath):
        if verbose:
            print(f">>> [load_jsonl] {filepath}")
        with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
            for line in fp_in:
                results.append(json.loads(line))
    else:
        if verbose:
            print(f">>> [load_jsonl] filepath does not exist: {filepath}")
    return results


def load_parquet(
        filepath: str,
        engine="auto",  # "auto", "pyarrow", "fastparquet"
        # mode: str = "r",
        # encoding: str = "utf-8",
        verbose: bool = False,
        # errors: Optional[str] = None,
) -> pd.DataFrame:
    results = []
    if os.path.isfile(filepath):
        if verbose:
            print(f">>> [load_parquet] {filepath}")
        # with open(filepath, mode, encoding=encoding, errors=errors) as fp_in:
        #     pass
        # Load a parquet object from the file path, returning a DataFrame.
        results = pd.read_parquet(filepath, engine=engine)
    else:
        if verbose:
            print(f">>> [load_parquet] filepath does not exist: {filepath}")
    return results


def main() -> None:
    timer_start = time.perf_counter()

    for eval_set, data_split in [("all", "train"), ("all", "test"), ("gem", "test")]:
        bench_data_dir = os.path.join("data/intent_grasp", eval_set)
        assert os.path.isdir(bench_data_dir), f"IntentGrasp evaluation data dir not found: {bench_data_dir}"

        # Load the metadata
        meta_filepath = os.path.join(bench_data_dir, f"metadata.json")
        assert os.path.isfile(meta_filepath)
        metadata = load_json(meta_filepath, verbose=False)
        assert isinstance(metadata, dict)

        # Load the data list
        eval_data_fp = os.path.join(bench_data_dir, f"{data_split}.jsonl")
        # eval_data_fp = os.path.join(bench_data_dir, f"{data_split}.parquet")
        assert os.path.isfile(eval_data_fp)
        eval_data = load_jsonl(eval_data_fp, verbose=False)
        # eval_data = load_parquet(eval_data_fp, verbose=False)
        assert isinstance(eval_data, list)

        # Check data types & format for each data item
        for data_item in eval_data:
            assert isinstance(data_item, dict) and len(data_item) == 8
            assert "id" in data_item and isinstance(data_item["id"], str) and len(data_item["id"]) > 0
            assert "speaker" in data_item and isinstance(data_item["speaker"], str) and len(data_item["speaker"]) > 0
            assert "context" in data_item and isinstance(data_item["context"], str) and len(data_item["context"]) > 0
            assert "question" in data_item and isinstance(data_item["question"], str) and len(data_item["question"]) > 0
            assert "options" in data_item and isinstance(data_item["options"], list)
            assert "answer_intent" in data_item and isinstance(data_item["answer_intent"], list)
            assert "answer_index" in data_item and isinstance(data_item["answer_index"], list)
            cur_options = data_item["options"]
            cur_answer_intent = data_item["answer_intent"]
            cur_answer_index = data_item["answer_index"]
            assert 1 <= len(cur_answer_intent) == len(cur_answer_index) <= len(cur_options) <= 10

            cur_options_set = set(cur_options)
            for _intent in cur_answer_intent:  # the correct intent answer must be in the options list
                assert isinstance(_intent, str) and _intent in cur_options_set

            assert "metadata" in data_item
            cur_metadata = data_item["metadata"]
            assert isinstance(cur_metadata, dict) and len(cur_metadata) == 9
            assert ("id" in cur_metadata and isinstance(cur_metadata["id"], str) and
                    cur_metadata["id"] == data_item["id"])
            assert ("paper_year" in cur_metadata and isinstance(cur_metadata["paper_year"], int) and
                    cur_metadata["paper_year"] > 0)
            assert ("original_task" in cur_metadata and isinstance(cur_metadata["original_task"], str) and
                    len(cur_metadata["original_task"]) > 0)
            assert ("original_split" in cur_metadata and isinstance(cur_metadata["original_split"], str) and
                    len(cur_metadata["original_split"]) > 0)
            assert ("text_form" in cur_metadata and isinstance(cur_metadata["text_form"], str) and
                    len(cur_metadata["text_form"]) > 0)
            assert ("intent_type" in cur_metadata and isinstance(cur_metadata["intent_type"], str) and
                    len(cur_metadata["intent_type"]) > 0)
            assert "is_synthetic" in cur_metadata and isinstance(cur_metadata["is_synthetic"], bool)
            assert "is_sensitive" in cur_metadata and isinstance(cur_metadata["is_sensitive"], bool)
            assert ("domain_topic" in cur_metadata and isinstance(cur_metadata["domain_topic"], list) and
                    len(cur_metadata["domain_topic"]) > 0)
            for d_t in cur_metadata["domain_topic"]:
                assert isinstance(d_t, list) and len(d_t) == 2
                assert isinstance(d_t[0], str) and len(d_t[0]) > 0
                assert isinstance(d_t[1], str)  # the topic can be empty

        print(f">>> Done Check: {bench_data_dir}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    print(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
