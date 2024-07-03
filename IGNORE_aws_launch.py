import sys
import os
import subprocess


# python aws_launch.py <master_addr> <master_port> <num_nodes> <rank>

# master_addr = sys.argv[1]
# master_port = sys.argv[2]
# num_nodes = sys.argv[3]
# rank = sys.argv[4]

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
