#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS=6
PORT=${PORT:-29500}
NAME=0427_linprobe0_imgnet

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../main_linprobe.py \
    --output_dir $(dirname "$0")/../ind_models/$NAME \
    --log_dir $(dirname "$0")/../ind_models/$NAME \
    --finetune $(dirname "$0")/../ind_models/mae_imagenet1k/checkpoint-199.pth \
    --nb_classes 2 \
    --accum_iter 7 \
    --batch_size 384 \
    --model vit_base_patch16 \
    --epochs 90 \
    --blr 0.1\
    --weight_decay 0.0 \
    --dist_eval \
    --img_prefix --data_path $(dirname "$0")/../ind_data/Bridge_Crack_Image/DBCC_Training_Data_Set \
    --local_rank 0
    # --resume $(dirname "$0")/../ind_models/$NAME/checkpoint-80.pth
