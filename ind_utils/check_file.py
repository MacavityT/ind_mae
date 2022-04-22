import os
import os.path as osp

root = '/mnt/VMSTORE/IndDatasets'
ann_file = osp.join(root, 'train.txt')
with open(ann_file, 'r') as f:
    samples = [osp.join(root, x.strip()) for x in f.readlines()]

for name in samples:
    if not os.path.exists(name):
        print(name)