#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import fire
import random
from typing import Optional

from tasks.utils_tasks import TASK_CLASS_DICT
from utils.data_io import DataIO
from utils.init_functions import logger_setup, cuda_setup, random_setup


def main(
        data_dir: Optional[str] = None,
        task_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        seed: int = 42,
        cuda: Optional[str] = None,
        verbose: bool = False,
        max_num_test_per_ds: int = 500,
        max_num_train_per_ds: int = 10000,
        **kwargs
) -> None:
    """
    :param data_dir: The directory to load/store the data.
    :param task_name: The name(s) of the source dataset(s). (e.g., "atis", "banking77", "atis,banking77", or "ALL")
    :param cache_dir: The root directory of the cache.
    :param project_root_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param max_num_test_per_ds: The maximum number of test instances for each source dataset to be used in IntentGrasp.
    :param max_num_train_per_ds: The max num of training instances for each source dataset to be used in IntentGrasp.

    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Build_IntentGrasp")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    # Project directory
    if not (isinstance(project_root_dir, str) and os.path.isdir(project_root_dir)):
        project_root_dir = os.getcwd()
    assert os.path.isdir(project_root_dir)

    # Data directory
    if not (isinstance(data_dir, str) and os.path.isdir(data_dir)):
        data_dir = os.path.join(project_root_dir, "data")
    assert os.path.isdir(data_dir), f"Root data dir not found: {data_dir}"
    raw_data_dir = os.path.join(data_dir, "raw_data")  # the directory to save all source datasets
    assert os.path.isdir(raw_data_dir), f"Raw data dir not found: {raw_data_dir}"

    if isinstance(task_name, str):
        if task_name == "ALL":
            task_name = list(TASK_CLASS_DICT.keys())
        else:
            task_name = [task_name]
    else:
        if isinstance(task_name, list) or isinstance(task_name, tuple):
            task_name = list(task_name)
        else:
            task_name = list(TASK_CLASS_DICT.keys())
    assert isinstance(task_name, list)

    # 1. unify the source datasets (with intent label contextualization and task reformatting)
    unified_data_dir = os.path.join(data_dir, "unified_data")  # the directory to save each processed source dataset
    os.makedirs(unified_data_dir, exist_ok=True)
    logger.info(f">>> <START> 1. Source datasets loading, parsing, and unification\n")
    for cur_task_name in task_name:
        cur_task_name = str(cur_task_name).strip()
        assert cur_task_name in TASK_CLASS_DICT
        eval_tasks = TASK_CLASS_DICT[cur_task_name](
            logger=logger, verbose=verbose, data_dir=data_dir,
            cache_dir=cache_dir, project_root_dir=project_root_dir)
        eval_tasks.load_task(do_save=True)
        logger.info(f">>> <Done> 1. {cur_task_name}")
    logger.info(f">>> <END> 1. Source datasets loading, parsing, and unification\n")

    # 2. Deduplicate & randomly downsample each source dataset (<=500 instances per dataset)
    logger.info(f">>> <START> 2. Randomly downsample the source datasets\n")
    all_metadata = dict()
    all_train, all_test = [], []
    all_text_set = set()
    all_num_dup = []
    for cur_task_idx, cur_task_name in enumerate(task_name):
        random.seed(seed + cur_task_idx)
        cur_task_train = []
        cur_task_test = []
        num_dup = [0]

        def _deduplicate_list(_data_list: list) -> list:
            _res_list = []
            for _data_item in _data_list:
                assert isinstance(_data_item, dict) and "context" in _data_item
                _data_item_text = str(_data_item["context"])
                if _data_item_text not in all_text_set:
                    all_text_set.add(_data_item_text)
                    _res_list.append(_data_item)
                else:
                    num_dup[0] += 1
            return _res_list

        cur_data_dir = os.path.join(unified_data_dir, cur_task_name)
        assert os.path.isdir(cur_data_dir)

        cur_meta_filepath = os.path.join(cur_data_dir, f"metadata.json")
        assert os.path.isfile(cur_meta_filepath)
        cur_metadata = DataIO.load_json(cur_meta_filepath, verbose=False)
        assert isinstance(cur_metadata, dict)
        all_metadata[cur_task_name] = cur_metadata

        data_train_filepath = os.path.join(cur_data_dir, f"train.jsonl")
        assert os.path.isfile(data_train_filepath)
        data_train = DataIO.load_jsonl(data_train_filepath, verbose=False)
        assert isinstance(data_train, list)
        cur_task_train.extend(data_train)

        data_valid_filepath = os.path.join(cur_data_dir, f"valid.jsonl")
        assert os.path.isfile(data_valid_filepath)
        data_valid = DataIO.load_jsonl(data_valid_filepath, verbose=False)
        assert isinstance(data_valid, list)
        cur_task_train.extend(data_valid)  # put the validation set of source datasets into IntentGrasp training set

        data_test_filepath = os.path.join(cur_data_dir, f"test.jsonl")
        assert os.path.isfile(data_test_filepath)
        data_test = DataIO.load_jsonl(data_test_filepath, verbose=False)
        assert isinstance(data_test, list)

        # Deduplication
        cur_task_test = _deduplicate_list(cur_task_test)  # Keep test instances first (if there are duplications)
        cur_task_train = _deduplicate_list(cur_task_train)
        all_num_dup.append(num_dup[0])

        # Random downsample the test set of each dataset
        if len(data_test) > max_num_test_per_ds:
            sample_idx_list = random.sample(range(len(data_test)), max_num_test_per_ds)
            sample_idx_set = set(sample_idx_list)
            sample_data = [_data for _d_idx, _data in enumerate(data_test) if _d_idx in sample_idx_set]
            left_data = [_data for _d_idx, _data in enumerate(data_test) if _d_idx not in sample_idx_set]
            assert len(sample_data) + len(left_data) == len(data_test)

            cur_task_test.extend(sample_data)  # keep at most `max_num_test_per_ds` test instances
            cur_task_train.extend(left_data)  # put the rest test instances into the training set
        else:
            cur_task_test.extend(data_test)
        assert len(cur_task_test) <= max_num_test_per_ds

        if len(cur_task_train) > max_num_train_per_ds:
            cur_task_train = random.sample(cur_task_train, max_num_train_per_ds)
        assert len(cur_task_train) <= max_num_train_per_ds

        all_train.extend(cur_task_train)
        all_test.extend(cur_task_test)
        logger.info(f">>> <Done> 2. {cur_task_name}")
    logger.info(f">>> <END> 2. Randomly downsample the source datasets\n")

    # 3. Marge and save the data
    bench_save_dir = os.path.join(data_dir, "intent_grasp_v1", "all")
    os.makedirs(bench_save_dir, exist_ok=True)
    DataIO.save_json(os.path.join(bench_save_dir, f"metadata.json"), all_metadata, mode="w", verbose=False, indent=2)
    DataIO.save_jsonl(os.path.join(bench_save_dir, f"train.jsonl"), all_train, mode="w", verbose=False)
    DataIO.save_jsonl(os.path.join(bench_save_dir, f"test.jsonl"), all_test, mode="w", verbose=False)
    DataIO.save_parquet(os.path.join(bench_save_dir, f"train.parquet"), all_train, verbose=False)
    DataIO.save_parquet(os.path.join(bench_save_dir, f"test.parquet"), all_test, verbose=False)
    logger.info(f">>> Done All. [# Train = {len(all_train)}] [# Test = {len(all_test)}] "
                f"bench_save_dir: {bench_save_dir}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
