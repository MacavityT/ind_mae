#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=4
PORT=${PORT:-29500}
NAME=debug

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../main_pretrain.py \
    --output_dir $(dirname "$0")/../ind_models/$NAME \
    --log_dir $(dirname "$0")/../ind_models/$NAME \
    --batch_size 24 \
    --accum_iter 28 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path  /data/share/imagenet/CLS-LOC/ \
    --local_rank 0 