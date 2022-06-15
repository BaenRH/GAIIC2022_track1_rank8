# _*_ coding: utf-8 _*_
"""
Time:     2022/4/11 19:09
Author:   Cai Ruihan
File:     test_kfold.py
"""
import os
import argparse
import json
from tqdm import tqdm
from datasets import PredDataset, process_pred_data, clear
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model_nezha import NezhaFinetune, NeZhaForJointLSTM, NeZhawithAttn


def main(opt):
    # 加载所有需要匹配的类别
    label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
    # 标签转化为对应id
    label2id = {key: i for i, key in enumerate(label_list)}

    # 文件路径
    # TODO:切换B榜
    pred_path = opt.data_root

    img_names, texts, img_features, queries = process_pred_data(pred_path)
    texts = list(map(clear, texts))

    print('总预测的图片数量：', len(img_names))

    tokenizer = BertTokenizer.from_pretrained(os.path.join(opt.pretrain_dir, f'{opt.kfold}_target_{opt.target}_kfold_0'))
    # 创建数据集
    dataset = PredDataset(
        img_names,
        tokenizer,
        texts,
        img_features,
        queries,
        label2id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 加载预训练好的权重
    if opt.model == 0:
        print('use NezhaFinetune')
        models = [NezhaFinetune.from_pretrained(
            os.path.join(opt.pretrain_dir, f'{opt.kfold}_target_{opt.target}_kfold_{kfold_idx}'))
                  for kfold_idx in range(opt.kfold)]
    elif opt.model == 1:
        print('use NeZhaForJointLSTM')
        models = [NeZhaForJointLSTM.from_pretrained(
            os.path.join(opt.pretrain_dir, f'{opt.kfold}_target_{opt.target}_kfold_{kfold_idx}'))
                  for kfold_idx in range(opt.kfold)]
    elif opt.model == 2:
        print('use NeZhawithAttn')
        models = [NeZhawithAttn.from_pretrained(
            os.path.join(opt.pretrain_dir, f'{opt.kfold}_target_{opt.target}_kfold_{kfold_idx}'))
                  for kfold_idx in range(opt.kfold)]

    for model in models:
        model.to(device)
        model.eval()

    preds = []
    with torch.no_grad():
        for i_b, batch in tqdm(enumerate(dataloader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            visual_embeds = batch['visual_embeds'].to(device)
            visual_attention_mask = batch['visual_attention_mask'].to(device)
            visual_token_type_ids = batch['visual_token_type_ids'].to(device)

            logits = []
            for model in models:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                unmasked_patch_features=visual_embeds,
                                visual_mask=visual_attention_mask,
                                visual_token_type_id=visual_token_type_ids,
                                )
                logits.append(torch.sigmoid(outputs))

            logits = sum(logits) / len(logits)

            for i in range(len(input_ids)):
                img_name = dataset.img_names[i_b * opt.batch_size + i]  # 提取人名
                query = dataset.queries[i_b * opt.batch_size + i]  # 提取要求的key
                match = {}
                for q in query:
                    match_id = label2id[q]
                    if opt.p == 1:
                        match[q] = logits[i, match_id].item()
                    else:
                        match[q] = 1 if logits[i, match_id] >= 0.5 else 0
                preds.append({"img_name": img_name, "match": match})

    print('总的预测图片数量:', len(preds))
    os.makedirs(opt.output_dir, exist_ok=True)
    pred_str = ''
    for pred in preds:
        pred_str += json.dumps(pred, ensure_ascii=False) + '\n'
    with open(os.path.join(opt.output_dir, f'pred_{opt.kfold}fold.txt'), 'w', encoding='utf-8') as f:
        f.write(pred_str)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../../data/tmp_data/logits', help='保存路径')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

    # parser.add_argument('--pretrain_dir', type=str, default='../../data/model_data/NEZHA_finetune', help='预训练模型的目录')
    parser.add_argument('--pretrain_dir', type=str, default='../../data/best_model/NEZHA_2m', help='预训练模型的目录')
    parser.add_argument('--data_root', type=str, default='../../data/contest_data/preliminary_testB.txt', help='数据目录')
    parser.add_argument('--kfold', type=int, default=5, help='交叉验证次数')
    parser.add_argument('--target', type=int, default=0, help='训练图文还是属性值。 1:图文；0：属性')
    parser.add_argument('--p', type=int, default=0, help='输出概率还是预测值，0预测，1概率')
    parser.add_argument('--model', type=int, default=1, help='选择模型：'
                                                             '0 NezhaFinetune,'
                                                             '1 NeZhaForJointLSTM'
                                                             '2 NeZhawithAttn')
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
