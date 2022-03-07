import sys
import os
import subprocess

master_addr = sys.argv[1]
master_port = sys.argv[2]

# NOTE: change as needed
nodelist = [
"dev-dy-p3dn24xlarge-1",
"dev-st-p3dn24xlarge-1",
"dev-st-p3dn24xlarge-2",
"train-st-p3dn24xlarge-1",
"train-st-p3dn24xlarge-2",
"train-st-p3dn24xlarge-3",
"train-st-p3dn24xlarge-4",
"train-st-p3dn24xlarge-5",
]

# NOTE: change as needed
# Constraints:
# 1. micro-batch-size must > 1
# 2. num-micro-batches must > 1
vit_args = "\
        --num-attention-heads 80 \
        --hidden-size 10240 \
        --num-layers 96 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 8 \
        `# --num-gpus 8` \
        --global-batch-size 8 \
        `# --data-parallel-size 1` \
        `# --num-micro-batches 2` \
        --micro-batch-size 4 \
        --DDP-impl local \
        --no-contiguous-buffers-in-local-ddp \
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
    \
        --num-classes 1000 \
        --img-h 224 \
        --img-w 224 \
        --num-channels 3 \
        --patch-dim 14 \
        --seq-length 256 \
        --max-position-embeddings 256 \
        --fp16"

# def get_command_for_rank(node, rank, master_addr, vit_args):
#     command = f"""ssh {node} \"
# echo {rank}
# echo {master_addr}
# echo {vit_args}
# \"
# """
#     return command

def get_command_for_rank(node, rank, master_addr, master_port, vit_args):
    command = f"""
ssh {node} \"
. /fsx/users/willfeng/conda/etc/profile.d/conda.sh
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=/fsx/users/willfeng/conda/envs/torch-nightly/bin/python3:$PATH
cd /fsx/users/willfeng/Megatron-LM

/fsx/users/willfeng/conda/envs/torch-nightly/bin/python3 -m torch.distributed.launch \
--nproc_per_node 8 \
--nnodes 8 \
--node_rank {rank} \
--master_addr {master_addr} \
--master_port {master_port} \
pretrain_vit_dummy.py {vit_args}\"
"""
    return command

print("Launching on all machines...")

commands = [
    get_command_for_rank(node, rank, master_addr, master_port, vit_args) \
    for rank, node in enumerate(nodelist)
]
for command in commands:
    print(command)

processes = [subprocess.Popen(
    [command], shell=True # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
) for command in commands[1:]]
# Only print head node output
processes.append(
    subprocess.Popen([commands[0]], shell=True)
)
for process in processes:
    process.wait()
