#!/usr/bin/env bash
source /root/miniconda3/bin/activate
source activate mae
# conda info -e > /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/env.log
python /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/slurm_test.py \
    > /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/cuda_test.log


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python $(dirname "$0")/../submitit_pretrain.py \
#     --ngpus 6 \
#     --nodes 4 \
#     --batch_size 24 \
#     --accum_iter 7 \
#     --model mae_vit_base_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 200 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path $(dirname "$0")/../ind_data/
