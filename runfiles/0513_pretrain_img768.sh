#!/usr/bin/env bash
source /root/miniconda3/bin/activate
conda activate mae

NAME=${NAME:-0513_pretrain_img768}
GPUS=6
NNODES=${NNODES:-6}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/../main_pretrain.py \
    --output_dir $(dirname "$0")/../ind_models/$NAME \
    --log_dir $(dirname "$0")/../ind_models/$NAME \
    --input_size 768 \
    --batch_size 1 \
    --accum_iter 114 \
    --model mae_vit_base_img768_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path $(dirname "$0")/../ind_data/ \
    --resume $(dirname "$0")/../ind_models/$NAME/checkpoint-75.pth

# Training instruction (2 Nodes)
#   First machine: 
#   NNODES=6 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh ind_runfiles/0513_pretrain_img768.sh
#   Second machine: 
#   NNODES=6 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh ind_runfiles/0513_pretrain_img768.sh
