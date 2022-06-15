# _*_ coding: utf-8 _*_
import json
import os
from data_process import process_data

# 加载所有需要匹配的类别
label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型',
              '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
# 标签转化为对应id
label2id = {key: i for i, key in enumerate(label_list)}

# 获取属性的所有值
# with open(os.path.join('../datasets', 'train', 'attr_to_attrvals.json'), 'r', encoding='utf-8') as f:
#     attr_to_attrvals = json.loads(f.read())
with open(os.path.join('../../data', 'train', 'attr_to_attrvals.json'), 'r', encoding='utf-8') as f:
    attr_to_attrvals = json.loads(f.read())

# 对attr_to_attrvals进行排序
sorted_attr_to_attrvals = {}
for attr, values in attr_to_attrvals.items():
    sort_values = []
    for value in values:
        if '=' in value:
            for v in value.split('='):
                sort_values.append(v)
        else:
            sort_values.append(value)
    # 按照长度进行排序，
    # 这样可以使 超短裙、中长袖、超长款、中长裙、半高领、中长款，排在短裙、长袖、长款、高领前面
    # 避免匹配的时候错误匹配
    sort_values = sorted(sort_values, key=lambda x: len(x), reverse=True)
    sorted_attr_to_attrvals[attr] = sort_values

class_to_attrs = {
    '鞋': ['闭合方式', '鞋帮高度'],
    '靴': ['闭合方式', '鞋帮高度'],
    '裤': ['裤门襟', '裤型', '裤长'],
    '包': ['类别'],
    # '裙': ['袖长', '裙长', '领型'],
    '上衣类': ['领型', '衣长', '穿着方式', '袖长', '版型', '裙长']
}

# 文件路径
# coarse_train_path = os.path.join('../datasets', 'train', 'coarse_data.json')
coarse_train_path = os.path.join('../../sample_v2', 'train', 'train_coarse.txt')  # 调试

# 获取数据
# with open(coarse_train_path, 'r', encoding='utf-8') as f:
#     coarse_data = json.loads(f.read())

# coarse_texts, coarse_img_features, coarse_labels, coarse_label_masks = \
#     coarse_data['texts'], coarse_data['img_features'], coarse_data['labels'], coarse_data['label_masks']
coarse_img_names, coarse_texts, coarse_img_features, coarse_labels, coarse_label_masks, coarse_key_attrs \
    = process_data(coarse_train_path, label2id)


coarse_to_fine_texts = []
coarse_to_fine_img_features = []
coarse_to_fine_labels = []
coarse_to_fine_label_masks = []
coarse_to_fine_key_attrs = []
for i in range(len(coarse_texts)):
    label = coarse_labels[i]
    # 只将coarse data种图文匹配的数据转化为fine data
    if label[0] == 0:
        continue
    text = coarse_texts[i]
    img_feature = coarse_img_features[i]
    label_mask = coarse_label_masks[i]
    key_attr = {}
    is_cloth = True
    for word in ['鞋', '靴', '包']:
        if word in text:
            is_cloth = False
            for attr in class_to_attrs[word]:
                for value in sorted_attr_to_attrvals[attr]:
                    if value in text:
                        key_attr[attr] = value
                        break
            break
    # 有极少数裤字和裙字在一起出现在标题的坏例子
    # 薄款2021年冬季九分裤女装裙裤裤子,薄款2021年夏季九分裤女装裙裤休闲裤
    if '裤' in text and '裙' not in text:
        is_cloth = False
        for attr in class_to_attrs['裤']:
            for value in sorted_attr_to_attrvals[attr]:
                if value in text:
                    key_attr[attr] = value
                    break
    if is_cloth:
        for attr in class_to_attrs['上衣类']:
            for value in sorted_attr_to_attrvals[attr]:
                if value in text:
                    key_attr[attr] = value
                    break
    if len(key_attr) > 0:
        for attr in key_attr.keys():
            label[label2id[attr]] = 1
            label_mask[label2id[attr]] = 1

        coarse_to_fine_key_attrs.append(key_attr)
        coarse_to_fine_texts.append(text)
        coarse_to_fine_img_features.append(img_feature)
        coarse_to_fine_labels.append(label)
        coarse_to_fine_label_masks.append(label_mask)

coarse_to_fine_data = {
    'texts': coarse_to_fine_texts,
    'img_features': coarse_to_fine_img_features,
    'labels': coarse_to_fine_labels,
    'label_masks': coarse_to_fine_label_masks,
    'key_attrs': coarse_to_fine_key_attrs
}

coarse_to_fine_json = json.dumps(coarse_to_fine_data)

#
# with open('../datasets/train/coarse_to_fine_data.json', 'w', encoding='utf-8') as f:
#     f.write(coarse_to_fine_json)

with open('../../sample_v2/train/coarse_to_fine_sample.json', 'w', encoding='utf-8') as f:
    f.write(coarse_to_fine_json)

print('Finish transforming!!!!')