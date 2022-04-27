#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=6
PORT=${PORT:-29500}
NAME=0427_linprobe2_ind

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../main_linprobe.py \
    --output_dir $(dirname "$0")/../ind_models/$NAME \
    --log_dir $(dirname "$0")/../ind_models/$NAME \
    --finetune $(dirname "$0")/../ind_models/0422_pretrain/checkpoint-199.pth \
    --nb_classes 6 \
    --accum_iter 7 \
    --batch_size 384 \
    --model vit_base_patch16 \
    --epochs 90 \
    --blr 0.1 \
    --cls_token \
    --weight_decay 0.0 \
    --dist_eval \
    --data_path $(dirname "$0")/../ind_data/NEU_surface_defect_database/NEU-CLS \
    --local_rank 0
    # --resume $(dirname "$0")/../ind_models/$NAME/checkpoint-80.pth
