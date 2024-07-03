# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

from submitit.core.submission import process_job


def submitit_main_patched() -> None:
    parser = argparse.ArgumentParser(description="Run a job")
    parser.add_argument("--folder", type=str, help="Folder where the jobs are stored (in subfolder)")
    args, unknown = parser.parse_known_args()
    process_job(args.folder)


if __name__ == "__main__":
    # This script is called by Executor.submit
    submitit_main_patched()
