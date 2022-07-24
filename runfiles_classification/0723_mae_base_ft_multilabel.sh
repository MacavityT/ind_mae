#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=6
PORT=${PORT:-29500}
NAME="$(basename $0 .sh)"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../main_finetune_sewer.py \
    --output_dir $(dirname "$0")/../ind_models/$NAME \
    --log_dir $(dirname "$0")/../ind_models/$NAME \
    --finetune $(dirname "$0")/../ind_models/mae_imagenet1k/mae_pretrain_vit_base.pth \
    --ds_mode e2e \
    --nb_classes 17 \
    --accum_iter 7 \
    --batch_size 24 \
    --input_size 224 \
    --model vit_base_patch16 \
    --epochs 12 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
    --dist_eval \
    --data_path $(dirname "$0")/../ind_data/SewerML 
    # --resume $(dirname "$0")/../ind_models/$NAME/checkpoint-0.pth