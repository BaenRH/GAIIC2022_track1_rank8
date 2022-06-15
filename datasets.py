import json
import random
import torch
import re
import numpy as np
import difflib


# 处理数据
def process_data(path, label2id):
    texts = []
    labels = []
    img_features = []
    label_masks = []
    key_attrs = []
    img_names = []

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

        # label2id 中key中的value值对应其位置的索引
        label = [0 for _ in range(len(label2id))]
        for match in matchs.keys():
            label[label2id[match]] = matchs[match]

        # 打掩码 用于掩去没有出现的key
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


def clear(x):
    # x = re.sub(r'[纯白红蓝粉黑]色','',x)
    x = re.sub(r'色', '', x)
    x = re.sub(r'\d+年', '', x)
    return x


def process_test_data(path, label2id):
    texts = []
    img_features = []
    label_masks = []
    img_names = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        """
        'img_name', 'title', 'key_attr', 'match', 'feature'
        """
        img_name = data['img_name']
        title = data['title']
        query = data['query']
        feature = data['feature']

        # 打掩码 用于掩去没有出现的key
        label_mask = [0 for _ in range(len(label2id))]
        for q in query:
            label_mask[label2id[q]] = 1

        img_names.append(img_name)
        texts.append(title)
        img_features.append(feature)
        label_masks.append(label_mask)

    return img_names, texts, img_features, label_masks


# 构造预训练的数据集
class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            texts,  # 文本
            visual_embeds,  # 图片特征
            labels,  # 标签
            key_attrs,  # 关键属性
            key_attr_values,  # 属性对应的所有值
            is_sentence_image=True,  # 是否执行图文匹配任务
            p1=0.4,  # 无关键属性数据改变title的概率
            p2=0.5,  # 有关键属性数据改变策略
            p3=0.5,  # 改变title策略的概率，p<p2, 直接改变整个标题，否则多属性随机改变
            p4=1,
            max_len=36,  # title按character的最大长度，原始数据集的title最大长度是36，新增关键属性文本的最大长度是30
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.visual_embeds = visual_embeds
        self.labels = labels
        self.key_attrs = key_attrs
        self.key_attr_values = key_attr_values
        # 是否执行sentence-image match prediction (classification)
        self.is_sentence_image = is_sentence_image
        # 改变title的概率，在self.is_sentence_image为True的情况下
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        """
        第一版，只输入一个句子
        """
        # visualbert模型的encoder的输入文本序列的最大长度：(这里只有一个句子)
        # max_len_texts + 2(special tokens [CLS][SEQ])
        self.max_len = max_len + 2
        """
        第二版，输入两个句子，第二个句子是关键属性构造成的
        """
        # self.max_len = max_len + 3

    def __getitem__(self, idx):

        visual_embeds = torch.tensor(self.visual_embeds[idx]).unsqueeze(0)
        # visual_token_type_ids 只有一种类型，所以设置为zeros
        visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        key_attrs = self.key_attrs[idx]

        # # 如果进行图文匹配任务，则进行数据增强手段，增加负样本
        # if self.is_sentence_image:

        # 无关键属性的数据
        if len(key_attrs) < 1:

            # 有一定概率改变整个title
            if random.random() < self.p1:
                text_idx = random.choice(range(len(self.labels)))
                while text_idx == idx:
                    text_idx = random.choice(range(len(self.labels)))
                sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
            # 不改变title
            else:
                text_idx = idx
                sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                   dtype=torch.long)

            text = self.texts[text_idx]
        else:
            # 一定概率不改变有关键属性的数据
            if random.random() < self.p2:
                text = self.texts[idx]
                sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                   dtype=torch.long)
            # 有一定概率改变整个title
            elif random.random() < self.p3:
                text_idx = random.choice(range(len(self.labels)))
                while text_idx == idx:
                    text_idx = random.choice(range(len(self.labels)))
                text = self.texts[text_idx]
                sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
            else:
                # 随机选取要改变的属性值数量
                random_num = random.choice(list(range(len(key_attrs))))
                random_idx_list = list(range(len(key_attrs)))
                random.shuffle(random_idx_list)
                random_idx_list = random_idx_list[:random_num + 1]
                keys = list(key_attrs.keys())
                text = self.texts[idx]

                # 交换位置
                if random.random() > self.p4:
                    attrs = list(key_attrs.values())
                    # 分离部位
                    ann = ''
                    for value in attrs:
                        ann += value + '|'
                    ann = ann[:-1]
                    # 切割
                    pieces = re.split(ann, text)
                    pieces = pieces + list(attrs)
                    # 重组
                    idxs = list(range(len(pieces)))
                    random.shuffle(idxs)
                    text = ''
                    for idx in idxs:
                        text += pieces[idx]

                for random_idx in random_idx_list:
                    random_key = keys[random_idx]
                    value = key_attrs[random_key]
                    random_value = random.choice(list(self.key_attr_values[random_key]))
                    while value in random_value:
                        random_value = random.choice(list(self.key_attr_values[random_key]))
                    if '=' in random_value:
                        random_value = random.choice(random_value.split('='))
                    text = text.replace(value, random_value)
                sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)

        # else:
        #     # text_idx = idx
        #     text = self.texts[idx]

        # text = self.texts[text_idx]

        inputs = self.tokenizer(text.lower(), padding="max_length", max_length=self.max_len, truncation=True)

        item = {key: torch.tensor(val) for key, val in inputs.items()}

        item.update({
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        })

        if self.is_sentence_image:
            item['sentence_image_labels'] = sentence_image_labels

        return item

    def __len__(self):
        return len(self.labels)


class FinetuneDateset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            texts,  # 文本
            labels,  # 标签，将图文匹配+12个属性匹配转化为13维的onehot编码
            visual_embeds,  # 图片特征，2048维
            label_masks,  # 标签掩码，如果Query存在相应的匹配任务，对应的label置1，不存在置0
            key_attrs,  # 关键属性，如果是无属性，则是空的
            key_attr_values,  # 属性对应的所有值
            label2id,  # match对应的id
            p_fine_pos=0.3,  # 关键属性数据作为正样本的概率，不做任何操作就是正样本，即图文和属性全匹配
            p_fine_title=0.3,  # 关键属性数据改变title的概率，即图文和属性全不匹配
            p_fine_attr=0.5,  # 关键属性数据改变attr的概率，即图文不匹配，属性部分不匹配或者全不匹配
            p_fine_same_key_attr=0.2,  # 关键属性数据替换具有相同的key attr的title的概率，即变为图文不匹配，属性全匹配
            # p2=0.5,             # 负样本数据增强操作选择直接替换整个title的概率
            p3=0.5,  # 对无关键属性的数据进行负样本增强操作的概率
            max_len=36,  # title按character的最大长度，原始数据集的title最大长度是36，新增关键属性文本的最大长度是30
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.visual_embeds = visual_embeds
        self.label_masks = label_masks
        self.key_attrs = key_attrs
        self.key_attr_values = key_attr_values
        self.label2id = label2id

        self.p_fine_pos = p_fine_pos
        self.p_fine_title = p_fine_title
        self.p_fine_attr = p_fine_attr
        self.p_fine_same_key_attr = p_fine_same_key_attr
        assert p_fine_title + p_fine_attr + p_fine_same_key_attr == 1

        # self.p2 = p2
        self.p3 = p3
        # visualbert模型的encoder的输入文本序列的最大长度：(这里只有一个句子)
        # max_len_texts + num_special tokens([CLS][SEQ]),一个句子两个special tokens，两个句子三个special tokens
        self.max_len = max_len + 2
        self.attrval_sameattr_values = get_sameattr_values(key_attr_values)

        # 相同含义的属性值
        same_mean_attrvals = []
        for values in key_attr_values.values():
            for value in values:
                if '=' in value:
                    same_mean_attrvals.append(value.split('='))
        self.same_mean_attrvals = same_mean_attrvals

        self.same_key_attr_idx = {}
        for idx, key_attr in enumerate(self.key_attrs):
            key_attr = {key: key_attr[key] for key in sorted(key_attr)}
            try:
                self.same_key_attr_idx[str(key_attr)].append(idx)
            except:
                self.same_key_attr_idx[str(key_attr)] = [idx]

    def __getitem__(self, idx):
        visual_embeds = torch.tensor(self.visual_embeds[idx], dtype=torch.float32).unsqueeze(0)
        # visual_token_type_ids 只有一种类型，所以设置为zeros
        visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        label_masks = torch.tensor(self.label_masks[idx])

        key_attrs = self.key_attrs[idx]

        # 如果是无关键属性数据，则不进行数据增强，直接返回原始数据
        # 新增处理二：无关键属性数据进行数据增强策略：直接改变title
        if len(key_attrs) < 1:
            """
            第一版代码：不对无关健属性数据进行任何的数据增强手段
            """
            """
            text = self.texts[idx]
            """
            # -------------------------------------------
            """
            第二版代码：对无关健属性数据做负样本数据增强操作，直接替换整个title
            """
            if random.random() < self.p3:
                text_idx = random.choice(range(len(self.labels)))
                while text_idx == idx:
                    text_idx = random.choice(range(len(self.labels)))
                text = self.texts[text_idx]
                labels = torch.zeros(13)
            else:
                text = self.texts[idx]

        # 如果是有关键属性数据，则进行数据增强策略来产生负样本
        else:
            # 一定概率不改变有关键属性的数据
            if random.random() < self.p_fine_pos:
                text = self.texts[idx]
            # 增加负样本的操作有两种，第一种直接替换整个title，
            # 第二种只改变某个属性值（可以多个，这里实现一个）
            else:
                neg_way = np.random.choice(['title', 'attr', 'same_key_attr'],
                                           p=[self.p_fine_title, self.p_fine_attr, self.p_fine_same_key_attr])
                # 直接替换整个title
                # if random.random() < self.p2:
                if neg_way == 'title':
                    text_idx = random.choice(range(len(self.labels)))
                    # 添加相似度，当两个文本相同或者是两个文本只在顺序不同，例如“修身型2021年冬季打底衫长袖常规款女装”和“常规款长袖女装2021年冬季修身型打底衫”
                    # 这两种情况下，他们的SequenceMatcher的相似度为1
                    sim = difflib.SequenceMatcher(None, self.texts[idx], self.texts[text_idx]).quick_ratio()
                    while text_idx == idx or sim == 1:
                        text_idx = random.choice(range(len(self.labels)))
                        sim = difflib.SequenceMatcher(None, self.texts[idx], self.texts[text_idx]).quick_ratio()
                    text = self.texts[text_idx]
                    labels = torch.zeros(13)

                    # 暂时默认全部不匹配
                    key_attrs = self.key_attrs[text_idx]
                    origin_key_attrs = self.key_attrs[idx]
                    for attr, value in origin_key_attrs.items():
                        if attr in key_attrs.keys():
                            # 随机更换title后，属性值相同
                            if value == key_attrs[attr]:
                                labels[self.label2id[attr]] = 1
                            # 随机更改title后，属性值具有相同含义
                            if is_same_mean_attrval(value, key_attrs[attr], self.same_mean_attrvals):
                                labels[self.label2id[attr]] = 1

                    label_masks = torch.tensor(self.label_masks[text_idx])

                    # mask = (label_masks + self.label_masks[text_idx]) > 0
                    # label_masks = torch.zeros_like(label_masks)
                    # label_masks[mask] = 1

                # 只改变某个属性值
                # else:
                elif neg_way == 'attr':
                    '''
                    第一版，只改变一种属性
                    '''
                    # random_key = random.choice(list(key_attrs.keys()))
                    # value = key_attrs[random_key]
                    # random_value = random.choice(list(self.key_attr_values[random_key]))
                    # while value in random_value:
                    #     random_value = random.choice(list(self.key_attr_values[random_key]))
                    # if '=' in random_value:
                    #     random_value = random.choice(random_value.split('='))
                    # text = self.texts[idx].replace(value, random_value)
                    # # 图文匹配置0
                    # labels[0] = 0
                    # # 随机选取属性匹配置0
                    # labels[self.label2id[random_key]] = 0

                    """
                    第二版，改变属性的数量是随机的
                    """
                    # 随机选取要改变的属性值数量, 最大改变3个属性
                    random_num = random.choice(list(range(len(key_attrs))))
                    random_idx_list = list(range(len(key_attrs)))
                    random.shuffle(random_idx_list)
                    random_idx_list = random_idx_list[:random_num + 1]
                    keys = list(key_attrs.keys())
                    text = self.texts[idx]
                    for random_idx in random_idx_list:
                        random_key = keys[random_idx]
                        value = key_attrs[random_key]
                        # 这种写法有点问题，换种写法
                        # random_value = random.choice(list(self.key_attr_values[random_key]))
                        # while value in random_value:
                        #     random_value = random.choice(list(self.key_attr_values[random_key]))
                        # if '=' in random_value:
                        #     random_value = random.choice(random_value.split('='))
                        random_value = random.choice(list(self.attrval_sameattr_values[random_key][value]))
                        text = text.replace(value, random_value)
                        # 随机选取属性匹配置0
                        labels[self.label2id[random_key]] = 0
                    # 图文匹配置0
                    labels[0] = 0
                #
                else:
                    key_attrs = {key: key_attrs[key] for key in sorted(key_attrs)}
                    if len(self.same_key_attr_idx[str(key_attrs)]) > 1:
                        random_idx = random.choice(self.same_key_attr_idx[str(key_attrs)])
                        while random_idx == idx:
                            random_idx = random.choice(self.same_key_attr_idx[str(key_attrs)])
                        text = self.texts[random_idx]

                        # 为了防止写在上面出现死循环，故写在while外面
                        sim = difflib.SequenceMatcher(None, self.texts[idx], self.texts[random_idx]).quick_ratio()
                        if sim != 1:
                            # 只有图文变得不匹配
                            labels[0] = 0

                    else:
                        text = self.texts[idx]

        inputs = self.tokenizer(text.lower(), padding="max_length", max_length=self.max_len, truncation=True)
        # inputs = self.tokenizer(text)

        # if '鞋' in text or '靴' in text:
        #     token_type_id = 0
        #     visual_token_type_ids = torch.full(visual_embeds.shape[:-1], 0, dtype=torch.long)
        # elif '裤' in text:
        #     token_type_id = 1
        #     visual_token_type_ids = torch.full(visual_embeds.shape[:-1], 1, dtype=torch.long)
        # elif '包' in text:
        #     token_type_id = 2
        #     visual_token_type_ids = torch.full(visual_embeds.shape[:-1], 2, dtype=torch.long)
        # else:
        #     token_type_id = 3
        #     visual_token_type_ids = torch.full(visual_embeds.shape[:-1], 3, dtype=torch.long)
        # inputs['token_type_ids'] = [token_type_id for _ in range(len(inputs['token_type_ids']))]

        item = {key: torch.tensor(val) for key, val in inputs.items()}

        item.update({
            "labels": labels,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "label_masks": label_masks,
        })

        return item

    def __len__(self):
        return len(self.texts)


# 获取同一属性的其他不同含义的属性值
def get_sameattr_values(attr_to_attrvals):
    attrval_sameattr_values = {attr: {} for attr in attr_to_attrvals}
    for attr, values in attr_to_attrvals.items():
        for i in range(len(values)):
            value = values[i]
            sameattr_values = []
            for j in range(len(values)):
                if j == i:
                    continue
                sameattr_value = values[j]
                if '=' in sameattr_value:
                    for v in sameattr_value.split('='):
                        sameattr_values.append(v)
                else:
                    sameattr_values.append(sameattr_value)
            if '=' in value:
                for v in value.split('='):
                    attrval_sameattr_values[attr][v] = sameattr_values
            else:
                attrval_sameattr_values[attr][value] = sameattr_values
    return attrval_sameattr_values


def is_same_mean_attrval(value1, value2, same_mean_attrvals):
    is_same = False
    for values in same_mean_attrvals:
        if value1 in values and value2 in values:
            is_same = True
            break
    return is_same


class TestDateset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 names,
                 texts,  # title
                 label_masks,  # 标签掩码
                 visual_embs,  # 图片特征
                 max_len=36,  # 句长
                 ):
        self.names = names
        self.tokenizer = tokenizer
        self.texts = texts
        self.label_masks = label_masks
        self.visual_embs = visual_embs
        self.max_len = max_len + 2  # 2个标志位 TODO：这儿是否要加2

    def __getitem__(self, idx):
        visual_embeds = torch.tensor(self.visual_embs[idx], dtype=torch.float32).unsqueeze(0)
        visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        label_masks = torch.tensor(self.label_masks[idx])
        names = self.names[idx]
        text = self.texts[idx]
        inputs = self.tokenizer(text.lower(), padding="max_length", max_length=self.max_len, truncation=True)
        # inputs = self.tokenizer(text, padding='max_length', max_length=self.max_len)
        item = {key: torch.tensor(val) for key, val in inputs.items()}

        item.update({
            "names": names,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "label_masks": label_masks,
        })

        return item

    def __len__(self):
        return len(self.texts)



# 交叉验证
def process_pred_data(path):
    img_names = []
    texts = []
    img_features = []
    queries = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        """
        'img_name', 'title', 'query', 'feature'
        """
        img_name = data['img_name']
        title = data['title']
        query = data['query']
        feature = data['feature']

        img_names.append(img_name)
        texts.append(title)
        img_features.append(feature)
        queries.append(query)

    return img_names, texts, img_features, queries


# 构造做预测任务的数据集
class PredDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_names,
            tokenizer,
            texts,
            visual_embeds,
            queries,
            label2id,
            max_len=36,  # title按character的最大长度
    ):
        self.tokenizer = tokenizer
        self.img_names = img_names
        self.texts = texts
        self.visual_embeds = visual_embeds
        self.queries = queries
        self.label2id = label2id
        # visualbert模型的encoder的输入文本序列的最大长度：(这里只有一个句子)
        # max_len_texts + 2(special tokens [CLS][SEQ])
        self.max_len = max_len + 2

    def __getitem__(self, idx):
        visual_embeds = torch.tensor(self.visual_embeds[idx], dtype=torch.float32).unsqueeze(0)
        # visual_token_type_ids 只有一种类型，所以设置为zeros
        visual_token_type_ids = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs = self.tokenizer(self.texts[idx].lower(), padding="max_length", max_length=self.max_len, truncation=True)
        item = {key: torch.tensor(val) for key, val in inputs.items()}

        item.update({
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        })

        return item

    def __len__(self):
        return len(self.texts)