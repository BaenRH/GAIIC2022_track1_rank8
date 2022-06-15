# _*_ coding: utf-8 _*_
import json

# 加载所有需要匹配的类别
label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型',
              '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
# 标签转化为对应id
label2id = {key: i for i, key in enumerate(label_list)}


# 处理数据
def process_data(path, label2id):
    img_names = []
    texts = []
    labels = []
    img_features = []
    label_masks = []
    key_attrs = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        """
        'img_name', 'title', 'key_attr', 'match', 'feature'
        """
        img_name = data['img_name']
        title = data['title']
        key_attr = data['key_attr']
        matchs = data['match']
        feature = data['feature']

        label = [0 for _ in range(len(label2id))]
        for match in matchs.keys():
            label[label2id[match]] = matchs[match]

        label_mask = [0 for _ in range(len(label2id))]
        for match in matchs.keys():
            label_mask[label2id[match]] = 1

        img_names.append(img_name)
        texts.append(title)
        labels.append(label)
        img_features.append(feature)
        label_masks.append(label_mask)
        key_attrs.append(key_attr)

    return img_names, texts, img_features, labels, label_masks, key_attrs


# 文件路径
# coarse_train_path = '../datasets/train/train_coarse.txt'  # 1000 - don't have key attr
# fine_train_path = '../datasets/train/train_fine.txt'  # 500 - have key attr
coarse_train_path = '../../sample_v2/train_coarse.txt'  # 1000 - don't have key attr
fine_train_path = '../../sample_v2/train_fine.txt'  # 500 - have key attr

fine_img_names, fine_texts, fine_img_features, fine_labels, fine_label_masks, fine_key_attrs = process_data(fine_train_path, label2id)
coarse_img_names, coarse_texts, coarse_img_features, coarse_labels,coarse_label_masks, coarse_key_attrs = process_data(coarse_train_path, label2id)

fine_data = {}
fine_data['img_names'] = fine_img_names
fine_data['texts'] = fine_texts
fine_data['img_features'] = fine_img_features
fine_data['labels'] = fine_labels
fine_data['label_masks'] = fine_label_masks
fine_data['key_attrs'] = fine_key_attrs

fine_json = json.dumps(fine_data)
# with open('../datasets/train/fine_data.json', 'w', encoding='utf-8') as f:
#     f.write(fine_json)
with open('../../sample_v2/fine_data.json', 'w', encoding='utf-8') as f:
    f.write(fine_json)

coarse_data = {}
coarse_data['img_names'] = coarse_img_names
coarse_data['texts'] = coarse_texts
coarse_data['img_features'] = coarse_img_features
coarse_data['labels'] = coarse_labels
coarse_data['label_masks'] = coarse_label_masks
coarse_data['key_attrs'] = coarse_key_attrs

coarse_json = json.dumps(coarse_data)

with open('../../sample_v2/coarse_data.json', 'w', encoding='utf-8') as f:
    f.write(coarse_json[:len(coarse_json)//2])
    f.write(coarse_json[len(coarse_json)//2:])

print(coarse_img_names)