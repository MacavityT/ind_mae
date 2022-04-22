import os
import os.path as osp
from tqdm import trange

root = '/mnt/VMSTORE/IndDatasets'
ann_file = osp.join(root, 'train.txt')
with open(ann_file, 'r') as f:
    samples = [osp.join(root, x.strip()) for x in f.readlines()]
bar = trange(len(samples))
for name in samples:
    if not os.access(name, os.F_OK):
        print(f'Exists error: {name}')
    if not os.access(name, os.R_OK):
        print(f'Access error: {name}')
    bar.update()