import sys
import os
import subprocess


# python aws_launch.py <master_addr> <master_port> <num_nodes> <rank>

# master_addr = sys.argv[1]
# master_port = sys.argv[2]
# num_nodes = sys.argv[3]
# rank = sys.argv[4]

# # Constraints:
# # 1. micro-batch-size must > 1
# # 2. num-micro-batches must > 1
# run_args = "\
#         --num-attention-heads 8 \
#         --hidden-size 5120 \
#         --num-layers 8 \
#         --tensor-model-parallel-size 8 \
#         --pipeline-model-parallel-size 8 \
#         `# --num-gpus 64` \
#         --global-batch-size 8 \
#         `# --data-parallel-size 1` \
#         `# --num-micro-batches 2` \
#         --micro-batch-size 4 \
#         --DDP-impl local \
#         --no-contiguous-buffers-in-local-ddp \
#         --activations-checkpoint-method uniform \
#         --distribute-checkpointed-activations \
#         --empty-unused-memory-level 2 \
#     \
#         --train-iters 10 \
#         --lr-decay-iters 320000 \
#         --data-impl mmap \
#         --split 949,50,1 \
#         --lr 0.00015 \
#         --lr-decay-style cosine \
#         --min-lr 1.0e-5 \
#         --weight-decay 1e-2 \
#         --clip-grad 1.0 \
#         --lr-warmup-fraction .01 \
#         --log-interval 1 \
#         --save-interval 10000 \
#         --eval-interval 1000 \
#         --eval-iters 1 \
#         --distributed-backend nccl \
#     \
#         --num-classes 1000 \
#         --img-h 224 \
#         --img-w 224 \
#         --num-channels 3 \
#         --patch-dim 14 \
#         --seq-length 256 \
#         --max-position-embeddings 256 \
#         --fp16"

# BERT-120B, compute in fp16, master weight copy in fp32
run_args="\
        --num-attention-heads 80 \
        --hidden-size 10240 \
        --num-layers 96 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 8 \
        `# --num-gpus 128` \
        --global-batch-size 2048 \
        `# --data-parallel-size 2` \
        `# --num-micro-batches 64` \
        --micro-batch-size 32 \
        --DDP-impl local \
        --activations-checkpoint-method uniform \
        --accumulate-allreduce-grads-in-fp32 \
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

# # BERT-10B
# run_args = "\
#         --num-attention-heads 32 \
#         --hidden-size 5120 \
#         --num-layers 32 \
#         --tensor-model-parallel-size 8 \
#         --pipeline-model-parallel-size 1 \
#         `# --num-gpus 32` \
#         --global-batch-size 2048 \
#         `# --data-parallel-size 4` \
#         `# --num-micro-batches 64` \
#         --micro-batch-size 8 \
#         --DDP-impl local \
#         --no-contiguous-buffers-in-local-ddp \
#         `# --activations-checkpoint-method uniform` \
#         `# --distribute-checkpointed-activations` \
#         `# --empty-unused-memory-level 2` \
#     \
#         --train-iters 10 \
#         --lr-decay-iters 320000 \
#         --data-impl mmap \
#         --split 949,50,1 \
#         --lr 0.00015 \
#         --lr-decay-style cosine \
#         --min-lr 1.0e-5 \
#         --weight-decay 1e-2 \
#         --clip-grad 1.0 \
#         --lr-warmup-fraction .01 \
#         --log-interval 1 \
#         --save-interval 10000 \
#         --eval-interval 1000 \
#         --eval-iters 1 \
#         --distributed-backend nccl \
#         --bert-no-binary-head \
#     \
#         --seq-length 256 \
#         --padded-vocab-size 256 \
#         --max-position-embeddings 256 \
#         --fp16"


export HOME_DIR=/fsx-mae/$USER  # FAIR AWS cluster
conda deactivate && conda deactivate && conda deactivate
. /fsx-mae/willfeng/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate && conda deactivate
conda activate dino_env_2022_03_15_py39_pt111_cu113
export MODULEPATH=/data/home/vkhalidov/modulefiles:$MODULEPATH
module unload cuda/11.4
module unload nccl/2.11.4-cuda.11.4
module unload nccl_efa/1.1.4-nccl.2.11.4-cuda.11.4
module load cuda/11.3 nccl/2.9.9-cuda.11.3 nccl_efa/1.1.4-nccl.2.9.9-cuda.11.3
export CUDA_HOME=/usr/local/cuda-11.3
source /data/home/vkhalidov/setup_efa.sh
cd /fsx-mae/willfeng/Megatron-LM
# Done env setup

export master_addr=a100-st-p4d24xlarge-2
export master_port=12340
export num_nodes=16
export python_file_name="pretrain_bert_dummy.py"
export rank=0

python3 -m torch.distributed.launch \
--nproc_per_node 8 \
--nnodes ${num_nodes} \
--node_rank ${rank} \
--master_addr ${master_addr} \
--master_port ${master_port} \
${python_file_name} ${run_args}

# def get_command_for_rank(rank, num_nodes, master_addr, master_port, python_file_name, run_args):
#     command = f"""
# export HOME_DIR=/fsx-mae/$USER  # FAIR AWS cluster
# conda deactivate && conda deactivate && conda deactivate
# . /fsx-mae/willfeng/miniconda3/etc/profile.d/conda.sh
# export PATH="/fsx-mae/willfeng/miniconda3/bin:$PATH"
# export PATH=/fsx-mae/willfeng/miniconda3/bin/python3:$PATH
# conda deactivate && conda deactivate && conda deactivate
# conda activate dino_env_2022_03_15_py39_pt111_cu113
# export MODULEPATH=/data/home/vkhalidov/modulefiles:$MODULEPATH
# module unload cuda/11.4
# module unload nccl/2.11.4-cuda.11.4
# module unload nccl_efa/1.1.4-nccl.2.11.4-cuda.11.4
# module load cuda/11.3 nccl/2.9.9-cuda.11.3 nccl_efa/1.1.4-nccl.2.9.9-cuda.11.3
# export CUDA_HOME=/usr/local/cuda-11.3
# source /data/home/vkhalidov/setup_efa.sh

# cd /fsx-mae/willfeng/Megatron-LM

# /fsx-mae/willfeng/miniconda3/bin/python3 -m torch.distributed.launch \
# --nproc_per_node 8 \
# --nnodes {num_nodes} \
# --node_rank {rank} \
# --master_addr {master_addr} \
# --master_port {master_port} \
# {python_file_name} {run_args}
# """
#     return command

# # os.system(get_command_for_rank(rank, num_nodes, master_addr, master_port, "pretrain_vit_dummy.py", run_args))
# # os.system(get_command_for_rank(rank, num_nodes, master_addr, master_port, "pretrain_gpt_dummy.py", run_args))
# os.system(get_command_for_rank(rank, num_nodes, master_addr, master_port, "pretrain_bert_dummy.py", run_args))
