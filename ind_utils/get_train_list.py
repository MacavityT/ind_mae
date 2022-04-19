from genericpath import isfile
import os
import os.path as osp

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif",
                  ".tiff", ".webp")
EXCLUDE_DATASET = ('A_解压完成压缩包', '$RECYCLE.BIN',
                   'UCI_Steel_Plates_Faults_Data_Set')
EXCLUDE_DATASET_FOLDER = ('GroundTruth', 'Label', 'Mask_images',
                          'ground_truth')

root = '/mnt/tmp'
txt = 'train.txt'
output = osp.join(root, txt)

datasets = os.listdir(root)
output_list = []
for dataset in datasets:
    if dataset in EXCLUDE_DATASET:
        continue
    dataset_path = osp.join(root, dataset)
    if osp.isfile(dataset_path):
        continue

    # get all image name and write in .txt
    for home, dirs, files in os.walk(dataset_path):
        # check folder name
        folder = home.split('/')[-1]
        if folder in EXCLUDE_DATASET_FOLDER:
            continue

        for filename in files:
            real_path = os.path.join(home, filename)
            save_path = real_path[len(root):]
            if save_path[0] == '/':
                save_path = save_path[1:]

            # remove some label picture
            if save_path.split('/')[0] == 'kolektor缺陷数据集':
                img_extensions = (".jpg", ".jpeg", ".png", ".ppm", ".pgm",
                                  ".tif", ".tiff", ".webp")  # remove .bmp
            elif save_path.split('/')[0] in ('磁瓦缺陷数据集',
                                             'Magnetic-Tile-Defect'):
                img_extensions = (".jpg", ".jpeg", ".ppm", ".pgm", ".tif",
                                  ".tiff", ".webp")  # remove .png
            else:
                img_extensions = IMG_EXTENSIONS

            # get file suffix
            suffix = os.path.splitext(filename)[-1]
            if suffix in img_extensions:
                output_list.append(save_path)

with open(output, 'w') as f:
    for name in output_list:
        f.write(name + '\n')