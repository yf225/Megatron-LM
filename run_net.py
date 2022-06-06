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
BERT-10B:

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
"""


"""
BERT-25B:

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
"""

"""
BERT-60B:

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
"""

"""
BERT-120B:

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
"""


"""
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
NUM_GPUS=128
MODEL_NAME=bert_25B

RUN_ARGS="\
        --num-attention-heads 32 \
        --hidden-size 7680 \
        --num-layers 36 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 4 \
        `# --num-gpus 128` \
        --global-batch-size 4096 \
        `# --data-parallel-size 4` \
        `# --num-micro-batches 64` \
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

NUM_GPUS = int(sys.argv[1])
MODEL_NAME = sys.argv[2]

def main():
    dist.multi_proc_run(num_proc=NUM_GPUS, model_name=MODEL_NAME, fun=pretrain_bert_dummy.train_model, use_spawn=True)


if __name__ == "__main__":
    main()
