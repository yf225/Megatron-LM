#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed helpers."""

import functools
import logging
import os
import pickle
import random
from datetime import datetime
import sys
import shlex

import numpy as np
import submitit
import torch
import argparse


def submitit_main_patched() -> None:
    parser = argparse.ArgumentParser(description="Run a job")
    parser.add_argument("--folder", type=str, help="Folder where the jobs are stored (in subfolder)")
    args, unknown = parser.parse_known_args()
    process_job(args.folder)
from submitit.core import submission
submission.submitit_main = submitit_main_patched


def _submitit_command_str_patched(self) -> str:
    return " ".join(
        [
            shlex.quote(sys.executable),
            "-u -m submitit.core._submit",
            "--folder",
            shlex.quote(str(self.folder)),
            *sys.argv[3:],  # NOTE: here we assume that Megatron-specific arguments starts at 3rd argument in the original shell command
        ]
    )
setattr(submitit.SlurmExecutor, '_submitit_command_str', property(_submitit_command_str_patched))


# Make work w recent PyTorch versions (https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_THREADING_LAYER"] = "GNU"


# TODO: need to run tests to ensure the changes in this file is correct.


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, fun):
        self.port = port
        self.fun = fun

    def __call__(self):
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        self.fun()


def single_proc_run(local_rank, fun, main_port, world_size):
    """Executes fun() on a single GPU in a multi-GPU setup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    fun()


def multi_proc_run(num_proc, model_name, fun, use_spawn=False):
    """Run a single or multi GPU job locally on the current node."""
    MODEL_NAME = model_name
    MODE = "slurm"
    MAX_RETRY = 3
    NAME = "megatron_job"
    OUT_DIR = f"{os.environ['HOME']}/checkpoints/{os.environ['USER']}/megatron_tp_pp/{MODEL_NAME}"
    PORT_RANGE = [10000, 65000]
    NUM_GPUS = num_proc

    # Launch fun() using submitit on SLURM
    use_slurm = MODE == "slurm"
    executor = submitit.AutoExecutor if use_slurm else submitit.LocalExecutor
    kwargs = {"slurm_max_num_timeout": MAX_RETRY} if use_slurm else {}
    executor = executor(folder=OUT_DIR, **kwargs)
    setup_executor(executor, NAME, NUM_GPUS)
    master_port = random.randint(PORT_RANGE[0], PORT_RANGE[1])
    job = executor.submit(SubmititRunner(master_port, fun))
    print(f"Submitted job_id {job.job_id} with out_dir: {OUT_DIR}/{job.job_id}")
    if not use_slurm:
        job.wait()


def setup_executor(executor, name, num_gpus, **kwargs):
    NUM_GPUS = num_gpus
    MAX_GPUS_PER_NODE = 8

    num_gpus_per_node = min(NUM_GPUS, MAX_GPUS_PER_NODE)
    exclude_nodes = "a100-st-p4d24xlarge-45,a100-st-p4d24xlarge-6,a100-st-p4d24xlarge-45,a100-st-p4d24xlarge-46,a100-st-p4d24xlarge-47,a100-st-p4d24xlarge-48"
    slurm_additional_parameters = {"mail-user": "", "mail-type": "END"}
    args = {
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,
        "cpus_per_task": 10,
        "nodes": max(1, NUM_GPUS // MAX_GPUS_PER_NODE),
        "timeout_min": 4200,
        "name": name,
        "slurm_partition": "scavenge", # learnfair
        "slurm_comment": "",
        "slurm_additional_parameters": slurm_additional_parameters,
    }
    # Comment out the following if on AWS cluster. There is no need for gpu type and exclude nodes etc in AWS yet.
    # if "lab" in launch.PARTITION or "fair" in launch.PARTITION:
    #     args["slurm_constraint"] = launch.GPU_TYPE
    #     args["mem_gb"] = launch.MEM_PER_GPU * num_gpus_per_node  # this line corresponds to `--mem` in submitit script
    # else:
    #     args["slurm_additional_parameters"]["exclude"] = exclude_nodes
    executor.update_parameters(**args, **kwargs)
