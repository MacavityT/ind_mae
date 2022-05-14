import os
import os.path as osp
import numpy as np
import random

root = '/mnt/VMSTORE/workspace_ty/IndDatasets/NEU_surface_defect_database/NEU-CLS'
# root = '/data/taiyan/DATASETS/industry/NEU_surface_defect_database/NEU-CLS'
all_txt = 'all.txt'
train_txt = 'train.txt'
val_txt = 'val.txt'
output_all = osp.join(root, all_txt)
output_train = osp.join(root, train_txt)
output_val = osp.join(root, val_txt)
val_ratio = 0.2

cats = os.listdir(root)
cats.remove('Thumbs.db')
cats = [cat.split('_')[0] for cat in cats]
cats_trim = []
for cat in cats:
    if cat not in cats_trim:
        cats_trim.append(cat)
cat2label = {cat: i for i, cat in enumerate(cats_trim)}

img_names = os.listdir(root)
img_names.remove('Thumbs.db')
output_all_dict = {}
output_train_dict = {}
output_val_dict = {}

for name in img_names:
    real_path = os.path.join(root, name)
    label = cat2label[name.split('_')[0]]
    if str(label) not in output_all_dict.keys():
        output_all_dict[str(label)] = [name]
    else:
        output_all_dict[str(label)].append(name)

for key, value in output_all_dict.items():
    output_val_cls = random.sample(value, int(len(value) * val_ratio))
    output_train_cls = []
    for data in value:
        if data not in output_val_cls:
            output_train_cls.append(data)

    output_train_dict[key] = output_train_cls
    output_val_dict[key] = output_val_cls

with open(output_train, 'w') as f:
    for key, value in output_train_dict.items():
        for data in value:
            f.write(data + ' ' + key + '\n')
with open(output_val, 'w') as f:
    for key, value in output_val_dict.items():
        for data in value:
            f.write(data + ' ' + key + '\n')
with open(output_all, 'w') as f:
    for key, value in output_all_dict.items():
        for data in value:
            f.write(data + ' ' + key + '\n')