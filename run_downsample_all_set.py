#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import fire
import random
import shutil

from utils.data_io import DataIO
from utils.init_functions import random_setup


def main(
        seed: int = 42,
        num_sample_subsets: int = 3,
) -> None:
    """
    :param seed: Random seed of all modules.
    :param num_sample_subsets: The number of downsample subsets.

    :return: None.
    """

    timer_start = time.perf_counter()
    random_setup(seed=seed)

    all_test_fp = os.path.join("data/intent_grasp/all", "test.jsonl")
    gem_test_fp = os.path.join("data/intent_grasp/gem", "test.jsonl")
    assert os.path.isfile(all_test_fp) and os.path.isfile(gem_test_fp)
    all_test_data = DataIO.load_jsonl(all_test_fp, verbose=True)
    gem_test_data = DataIO.load_jsonl(gem_test_fp, verbose=True)

    assert isinstance(all_test_data, list) and len(all_test_data) > 0
    assert isinstance(gem_test_data, list) and len(gem_test_data) > 0
    num_all_test = len(all_test_data)
    num_gem_test = len(gem_test_data)
    print(f">>> [num_all_test = {num_all_test}] [num_gem_test = {num_gem_test}]")

    # Sample num_gem_test data items from all_test_data
    for sample_turn in range(num_sample_subsets):
        cur_seed = seed + sample_turn
        random.seed(cur_seed)
        cur_sample_data = random.sample(all_test_data, num_gem_test)
        assert isinstance(cur_sample_data, list) and len(cur_sample_data) == num_gem_test

        cur_save_dir = os.path.join("data/intent_grasp", f"all2gem_{cur_seed}")
        os.makedirs(cur_save_dir, exist_ok=True)

        try:
            shutil.copy(os.path.join("data/intent_grasp/all", "metadata.json"), cur_save_dir)
        except Exception as e:
            print(e)
            return None

        DataIO.save_jsonl(os.path.join(cur_save_dir, "test.jsonl"), cur_sample_data, mode="w", verbose=True)
        DataIO.save_parquet(os.path.join(cur_save_dir, "test.parquet"), cur_sample_data, verbose=True)
        print(f">>> Done saving. cur_save_dir: {cur_save_dir}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    print(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
