#!/usr/bin/env bash
source /root/miniconda3/bin/activate
source activate mae

python /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/cuda_test.py \
    > /mnt/VMSTORE/workspace_ty/ind_mae/ind_runfiles/test.log