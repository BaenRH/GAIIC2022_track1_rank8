# _*_ coding: utf-8 _*_
"""
Time:     2022/3/22 15:07
Author:   Cai Ruihan
File:     test.py.py
"""
import torch
from datasets import *
from model_nezha import NezhaFinetune, NeZhaForJointLSTM,NeZhawithAttn, NeZhaFuse
import numpy as np
from transformers import BertTokenizer
import os
import argparse
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='../../best_model', help='保存路径')  # TODO
parser.add_argument('--pretrain_dir', type=str, default='../save_pretrain', help='预训练模型的目录')
parser.add_argument('--data_root', type=str, default='../sample_v2', help='数据目录')

parser.add_argument('--test_dir', type=str, default='../sample_v2', help='测试目录')  # 5.22
parser.add_argument('--output_path', type=str, default='../sample_v2', help='测试输出目录')
# config
parser.add_argument('--batch_size', type=int, default=360, help='batch_size')
parser.add_argument('--lr', type=float, default=5e-4, help='lr')
parser.add_argument('--p', type=int, default=0, help='输出概率还是预测值，0预测，1概率')
parser.add_argument('--model', type=int, default=1, help='选择模型：'
                                                         '0 NezhaFinetune,'
                                                         '1 NeZhaForJointLSTM'
                                                         '2 NeZhawithAttn'
                                                         '3 NezhaFuse')

opt = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(os.path.join(opt.output_dir, f'target_2_{opt.model}'))


# 加载所有需要匹配的类别  并 标签转化为对应id
label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
# 标签转化为对应id
label2id = {key: i for i, key in enumerate(label_list)}

# 修改路径
# TODO
test_dir = os.path.join('../..',os.path.split(opt.test_dir)[-1])
output_path = os.path.join('../..'+opt.output_path,'test_nezha.txt')
print('test dir:',test_dir)
print('output_path:',output_path)

# 读取数据
img_names, texts, img_features, label_masks = \
    process_test_data(test_dir, label2id)
print('finish loading test data')


# 4.1 加clear
texts = list(map(clear, texts))

dataset = TestDateset(
    tokenizer=tokenizer,
    names=img_names,
    texts=texts,
    label_masks=label_masks,
    visual_embs=img_features,
)

dataloader = DataLoader(dataset, batch_size=opt.batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型选择
if opt.model == 0:
    model = NezhaFinetune.from_pretrained(pretrained_model_name_or_path=os.path.join(opt.output_dir, f'target_2_{opt.model}'))
elif opt.model == 1:
    model = NeZhaForJointLSTM.from_pretrained(pretrained_model_name_or_path=os.path.join(opt.output_dir, f'target_2_{opt.model}'))
elif opt.model == 2:
    model = NeZhawithAttn.from_pretrained(pretrained_model_name_or_path=os.path.join(opt.output_dir, f'target_2_{opt.model}'))
elif opt.model == 3:
    model = NeZhaFuse.from_pretrained(pretrained_model_name_or_path=os.path.join(opt.output_dir, f'target_2_{opt.model}'))
model.to(device=device)

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device=device)  # batchsize*text_length
        token_type_ids = batch['token_type_ids'].to(device=device)  # batchsize*text_length
        attention_mask = batch['attention_mask'].to(device=device)
        visual_embeds = batch['visual_embeds'].to(device=device)
        visual_mask = batch['visual_attention_mask'].to(device=device)
        visual_token_type_id = batch['visual_token_type_ids'].to(device=device)
        label_masks = batch['label_masks'].to(device=device)
        names = batch['names']
        model.eval()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            unmasked_patch_features=visual_embeds,
            visual_mask=visual_mask,
            visual_token_type_id=visual_token_type_id,
        )

        output = torch.sigmoid(output)
        label_masks = label_masks.cpu()
        if opt.p == 0:
            # TODO
            with open('output/test2.txt', 'a', encoding='utf-8') as f:
                for i in np.arange(len(names)):
                    name = names[i]
                    out1 = {"img_name": name}
                    out2 = {}
                    idx = np.where(label_masks[i, :] == 1)[0]  # 去掉图文
                    for j in idx:
                        kn = label_list[j]  # 获取对应属性值
                        out2[kn] = int(output[i, j].item() > 0.5)  # 属性值对应预测值
                    out1.update({"match": out2})

                    js = json.dumps(out1, ensure_ascii=False)
                    f.write(js)
                    f.write('\n')
        else:
            with open(output_path, 'a', encoding='utf-8') as f:
                for i in np.arange(len(names)):
                    name = names[i]
                    out1 = {"img_name": name}
                    out2 = {}
                    idx = np.where(label_masks[i, :] == 1)[0]  # 去掉图文
                    for j in idx:
                        kn = label_list[j]  # 获取对应属性值
                        out2[kn] = output[i, j].item()    # 属性值对应预测值
                    out1.update({"match": out2})

                    js = json.dumps(out1, ensure_ascii=False)
                    f.write(js)
                    f.write('\n')

    print('Finish testing')
