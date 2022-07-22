#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute various operations (train, test, time, etc.) on a classification model."""

import argparse
import sys

import distributed as dist
import pretrain_bert_dummy 


"""
FAIR AWS cluster:

Reference: https://fb.workplace.com/groups/aws.fair.discuss/posts/1009185456666472/

# FIRST RUN ONLY:
# 1. Rsync the `dino_env_2022_04_12_py39_pt111_cu114` env from /data/home/vkhalidov/miniconda/envs/dino_env_2022_04_12_py39_pt111_cu114 to your local conda
#   e.g. rsync -avr /data/home/vkhalidov/miniconda/envs/dino_env_2022_04_12_py39_pt111_cu114 <your_conda_path>/envs/dino_env_2022_04_12_py39_pt111_cu114
# 2. Install the apex library: https://github.com/NVIDIA/apex

conda activate dino_env_2022_04_12_py39_pt111_cu114
export MODULEPATH=/data/home/vkhalidov/modulefiles:$MODULEPATH
module unload cuda
module unload nccl
module unload nccl_efa
module load cuda/11.4 nccl/2.12.7-cuda.11.4 nccl_efa/1.2.0-nccl.2.12.7-cuda.11.4
export CUDA_HOME=/usr/local/cuda-11.4
source /data/home/vkhalidov/setup_efa.sh
"""


"""
AI AWS cluster:

conda activate dino_env_2022_04_12_py39_pt111_cu114

export CUDA_VER_SHORT=114
export CUDA_VER=11.4

module unload cuda
module unload nccl
module unload nccl_efa
module load cuda/${CUDA_VER}
module load nccl/2.12.7-cuda.${CUDA_VER}
module load nccl_efa/1.2.0-nccl.2.12.7-cuda.${CUDA_VER}

source /data/shared/bin/cluster_env_new.sh
source /data/home/willfeng/setup_efa.sh

export CUDA_HOME=/usr/local/cuda-${CUDA_VER}
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_PATH=${CUDA_HOME}
export CUDNN_INSTALL_PATH=${CUDA_HOME}
export NCCL_INCLUDE_DIR=${CUDA_HOME}/include
export NCCL_INCLUDE_DIRS=${CUDA_HOME}/include
export NCCL_ROOT_DIR=${CUDA_HOME}
export NCCL_LIB_DIR=${CUDA_HOME}/lib
export NCCL_LIBRARIES=${CUDA_HOME}/lib/libnccl.so.2.12.7
export NCCL_LIBRARY=${CUDA_HOME}/lib/libnccl.so.2.12.7
export USE_SYSTEM_NCCL=ON
"""


"""
BERT-10B:

NUM_GPUS=...
MODEL_NAME=bert_10B
RUN_ARGS="\
        --num-attention-heads 32 \
        --hidden-size 5120 \
        --num-layers 32 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        `# --num-gpus ${NUM_GPUS}` \
        --global-batch-size 4096 \
        `# --data-parallel-size 16` \
        `# --num-micro-batches 16` \
        --micro-batch-size 16 \
        --DDP-impl local \
        --accumulate-allreduce-grads-in-fp32 \
        `# --activations-checkpoint-method uniform` \
        `# --distribute-checkpointed-activations` \
        `# --empty-unused-memory-level 2` \
    \
        --train-iters 10 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --distributed-backend nccl \
        --bert-no-binary-head \
    \
        --seq-length 256 \
        --padded-vocab-size 256 \
        --max-position-embeddings 256 \
        --fp16"
./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
"""


"""
BERT-25B:

NUM_GPUS=...
MODEL_NAME=bert_25B
RUN_ARGS="\
        --num-attention-heads 32 \
        --hidden-size 7680 \
        --num-layers 36 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 2 \
        `# --num-gpus ${NUM_GPUS}` \
        --global-batch-size 2048 \
        `# --data-parallel-size 8` \
        `# --num-micro-batches 32` \
        --micro-batch-size 16 \
        --DDP-impl local \
        --accumulate-allreduce-grads-in-fp32 \
        `# --activations-checkpoint-method uniform` \
        `# --distribute-checkpointed-activations` \
        `# --empty-unused-memory-level 2` \
    \
        --train-iters 10 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --distributed-backend nccl \
        --bert-no-binary-head \
    \
        --seq-length 256 \
        --padded-vocab-size 256 \
        --max-position-embeddings 256 \
        --fp16"
./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
"""

"""
BERT-60B:

NUM_GPUS=...
MODEL_NAME=bert_60B
RUN_ARGS="\
        --num-attention-heads 32 \
        --hidden-size 10240 \
        --num-layers 48 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 2 \
        `# --num-gpus ${NUM_GPUS}` \
        --global-batch-size 2048 \
        `# --data-parallel-size 8` \
        `# --num-micro-batches 256` \
        --micro-batch-size 1 \
        --DDP-impl local \
        --accumulate-allreduce-grads-in-fp32 \
        `# --activations-checkpoint-method uniform` \
        `# --distribute-checkpointed-activations` \
        `# --empty-unused-memory-level 2` \
    \
        --train-iters 10 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --distributed-backend nccl \
        --bert-no-binary-head \
    \
        --seq-length 256 \
        --padded-vocab-size 256 \
        --max-position-embeddings 256 \
        --fp16"
./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
"""

"""
BERT-120B:

NUM_GPUS=...
MODEL_NAME=bert_120B
RUN_ARGS="\
        --num-attention-heads 80 \
        --hidden-size 10240 \
        --num-layers 96 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 8 \
        `# --num-gpus ${NUM_GPUS}` \
        --global-batch-size 2048 \
        `# --data-parallel-size 2` \
        `# --num-micro-batches 256` \
        --micro-batch-size 4 \
        --DDP-impl local \
        --activations-checkpoint-method uniform \
        --accumulate-allreduce-grads-in-fp32 \
        --distribute-checkpointed-activations \
        --empty-unused-memory-level 2 \
    \
        --train-iters 10 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --distributed-backend nccl \
        --bert-no-binary-head \
    \
        --seq-length 256 \
        --padded-vocab-size 256 \
        --max-position-embeddings 256 \
        --fp16"
./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
"""

"""
BERT-120B dummy optimizer (for Alpa comparison):

NUM_GPUS=64
MODEL_NAME=bert_120B
RUN_ARGS="\
        --optimizer dummy \
    \
        --num-attention-heads 80 \
        --hidden-size 10240 \
        --num-layers 96 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 8 \
        `# --num-gpus ${NUM_GPUS}` \
        --global-batch-size 512 \
        `# --data-parallel-size 1` \
        `# --num-micro-batches 128` \
        --micro-batch-size 4 \
        --DDP-impl local \
        --activations-checkpoint-method uniform \
        --distribute-checkpointed-activations \
        --empty-unused-memory-level 2 \
    \
        --train-iters 10 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --distributed-backend nccl \
        --bert-no-binary-head \
    \
        --seq-length 256 \
        --padded-vocab-size 256 \
        --max-position-embeddings 256 \
        --fp16"
./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
"""


"""
param sweep: 

NUM_GPUS=128
MODEL_NAME=bert_25B

for num_micro_batches in 32 16 8 4
do
RUN_ARGS="\
        --num-attention-heads 32 \
        --hidden-size 7680 \
        --num-layers 36 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 4 \
        `# --num-gpus ${NUM_GPUS}` \
        --global-batch-size 4096 \
        `# --data-parallel-size 4` \
        `# --num-micro-batches ${num_micro_batches}` \
        --micro-batch-size $((4096 / 4 / num_micro_batches)) \
        --DDP-impl local \
        --accumulate-allreduce-grads-in-fp32 \
        `# --activations-checkpoint-method uniform` \
        `# --distribute-checkpointed-activations` \
        `# --empty-unused-memory-level 2` \
    \
        --train-iters 10 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1 \
        --distributed-backend nccl \
        --bert-no-binary-head \
    \
        --seq-length 256 \
        --padded-vocab-size 256 \
        --max-position-embeddings 256 \
        --fp16"

./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
done
"""

"""
./run_net.py ${NUM_GPUS} ${MODEL_NAME} ${RUN_ARGS}
"""

NUM_GPUS = int(sys.argv[1])
MODEL_NAME = sys.argv[2]

def main():
    dist.multi_proc_run(num_proc=NUM_GPUS, model_name=MODEL_NAME, fun=pretrain_bert_dummy.train_model, use_spawn=True)


if __name__ == "__main__":
    main()
