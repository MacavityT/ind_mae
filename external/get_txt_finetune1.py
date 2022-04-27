import os
import os.path as osp
import numpy as np
import random

root = '/mnt/VMSTORE/workspace_ty/IndDatasets/Kylberg_Texture_Dataset_v.1.0/'
all_txt = 'all.txt'
train_txt = 'train.txt'
val_txt = 'val.txt'
output_all = osp.join(root, all_txt)
output_train = osp.join(root, train_txt)
output_val = osp.join(root, val_txt)
val_ratio = 0.2

cats = os.listdir(root)
cats = [cat.split('-')[0] for cat in cats]
cat2label = {cat: i for i, cat in enumerate(cats)}

folders = os.listdir(root)
output_all_list = []
output_train_list = []
output_val_list = []

for folder in folders:
    folderpath = osp.join(root, folder)
    img_names = os.listdir(folderpath)

    output_all_cls = []
    for name in img_names:
        real_path = os.path.join(folderpath, name)
        save_path = real_path[len(root):]
        if save_path[0] == '/':
            save_path = save_path[1:]
        output_all_cls.append(save_path)

    output_val_cls = random.sample(output_all_cls,
                                   int(len(output_all_cls) * val_ratio))
    output_train_cls = []
    for data in output_all_cls:
        if data not in output_val_cls:
            output_train_cls.append(data)

    output_train_list.append(output_train_cls)
    output_val_list.append(output_val_cls)
    output_all_list.append(output_all_cls)

with open(output_train, 'w') as f:
    for i, datas in enumerate(output_train_list):
        for data in datas:
            f.write(data + ' ' + str(i) + '\n')
with open(output_val, 'w') as f:
    for i, datas in enumerate(output_val_list):
        for data in datas:
            f.write(data + ' ' + str(i) + '\n')
with open(output_all, 'w') as f:
    for i, datas in enumerate(output_all_list):
        for data in datas:
            f.write(data + ' ' + str(i) + '\n')
