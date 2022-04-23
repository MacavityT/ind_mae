#!/usr/bin/env bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH  \
python /mnt/VMSTORE/workspace_ty/ind_mae/ind_utils/check_file.py > /mnt/VMSTORE/checkfile_log.log
sleep 1m
