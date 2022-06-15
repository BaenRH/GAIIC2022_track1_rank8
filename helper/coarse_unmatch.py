# _*_ coding: utf-8 _*_
"""
Time:     2022/4/5 11:07
Author:   Cai Ruihan
File:     coarse_numatch.py
Function: 用于提取coarse data 中图文不匹配的数据
"""
import json
import os
from data_process import process_data
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../input/track1_contest_4362/train/train/', help='数据目录')
opt = parser.parse_args()
print(opt)
data_dir = os.path.join('../../..',opt.data_dir)
print('文件路径：',data_dir)

# 加载所有需要匹配的类别
label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型',
              '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
# 标签转化为对应id
label2id = {key: i for i, key in enumerate(label_list)}

# 获取属性的所有值
coarse_train_path = os.path.join(data_dir, 'train_coarse.txt')  # TODO
coarse_img_names, coarse_texts, coarse_img_features, coarse_labels, coarse_label_masks, coarse_key_attrs \
    = process_data(coarse_train_path, label2id)

# 寻找不匹配的索引
idxs = np.where(np.array(coarse_labels)[:, 0] == 0)[0]
coarse_unmatch_texts = np.array(coarse_texts)[idxs].tolist()
coarse_unmatch_img_features = np.array(coarse_img_features)[idxs].tolist()
coarse_unmatch_labels = np.array(coarse_labels)[idxs].tolist()
coarse_unmatch_label_masks = np.array(coarse_label_masks)[idxs].tolist()
coarse_unmatch_key_attrs = np.array(coarse_key_attrs)[idxs].tolist()


print(len(coarse_img_names))
coarse_unmatch_data = {
    'texts': coarse_unmatch_texts,
    'img_features': coarse_unmatch_img_features,
    'labels': coarse_unmatch_labels,
    'label_masks': coarse_unmatch_label_masks,
    'key_attrs': coarse_unmatch_key_attrs
}

coarse_to_fine_json = json.dumps(coarse_unmatch_data)

# 训练数据  # TODO： 修改
with open('../coarse_unmatch_data.json', 'w', encoding='utf-8') as f:
    f.write(coarse_to_fine_json)

# sample数据
# with open('../../sample_v2/coarse_unmatch_sample.json', 'w', encoding='utf-8') as f:
#     f.write(coarse_to_fine_json)

print('Finish !!!!')
