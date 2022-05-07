#!/usr/bin/env bash
source /root/miniconda3/bin/activate
source activate mae

# python /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/slurm_test.py \
#     > /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/cuda_test.log

GPUS=6
PORT=${PORT:-29500}
NAME=slurm_test

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
    --data_path $(dirname "$0")/../ind_data/ \
    --local_rank 0 \
    > $(dirname "$0")/../ind_models/$NAME/shell_output.log
