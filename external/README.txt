1. check_file: 用于检查文件路径及可读写性是否正常
2. get_txt_finetune1: 生成Kylberg_Texture_Dataset_v.1.0数据集用于finetune的train/val文件
3. get_txt_finetune2: 生成NEU_surface_defect_database数据集用于finetune的train/val文件
4. get_txt_pretrain: 生成全部数据集的train.txt文件
5. slide_split_datasets: 多线程切分图像224*224
6. check_dataset: 用于检查数据集dataloader是否正常
7. img_similarity：检查爬虫数据爬取到的图片相似度