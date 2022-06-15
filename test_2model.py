# _*_ coding: utf-8 _*_
"""
Time:     2022/4/3 15:58
Author:   Cai Ruihan
File:     test_2model.py
"""
from datasets import *
from model_nezha import NezhaFinetune, NeZhaForJointLSTM, NeZhawithAttn
import numpy as np
from transformers import BertTokenizer
import os
import argparse
from torch.utils.data import DataLoader


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=360, help='batch_size')
    # parser.add_argument('--output_dir', type=str, default='../../data/model_data/NEZHA_finetune', help='保存路径')   # TODO
    parser.add_argument('--output_dir', type=str, default='../../data/best_model/NEZHA_2m', help='保存路径')
    parser.add_argument('--test_dir', type=str, default='../../data/contest_data/preliminary_testB.txt', help='测试集文件位置')  # TODO
    parser.add_argument('--p', type=int, default=0, help='输出概率还是预测值，0预测，1概率')
    parser.add_argument('--model', type=int, default=1, help='选择模型：'
                                                             '0 NezhaFinetune,'
                                                             '1 NeZhaForJointLSTM'
                                                             '2 NeZhawithAttn')
    # TODO : 不同模型不同target同时进行预测。
    opt = parser.parse_args()
    return opt


def main(opt):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(opt.output_dir, 'target_1'))     # TODO

    # 加载所有需要匹配的类别  并 标签转化为对应id
    label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
    # 标签转化为对应id
    label2id = {key: i for i, key in enumerate(label_list)}

    # 读取数据
    # TODO: 修改路径
    img_names, texts, img_features, label_masks = \
        process_test_data(opt.test_dir, label2id)

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

    # 选择模型
    # TODO： 加载不同模型
    # 加载预训练好的权重
    if opt.model == 0:
        print('use NezhaFinetune')
        model0 = NezhaFinetune.from_pretrained(
            pretrained_model_name_or_path=os.path.join(opt.output_dir, 'target_0'))  # 用于预测属性
        model1 = NezhaFinetune.from_pretrained(
            pretrained_model_name_or_path=os.path.join(opt.output_dir, 'target_1'))  # 用于预测图文
    elif opt.model == 1:
        print('use NeZhaForJointLSTM')
        model0 = NeZhaForJointLSTM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(opt.output_dir, 'target_0'))  # 用于预测属性
        model1 = NeZhaForJointLSTM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(opt.output_dir, 'target_1'))  # 用于预测图文
    elif opt.model == 2:
        print('use NeZhawithAttn')
        model0 = NeZhawithAttn.from_pretrained(
            pretrained_model_name_or_path=os.path.join(opt.output_dir, 'target_0'))  # 用于预测属性
        model1 = NeZhawithAttn.from_pretrained(
            pretrained_model_name_or_path=os.path.join(opt.output_dir, 'target_1'))  # 用于预测图文

    model0.to(device=device)
    model1.to(device=device)

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

            model0.eval()
            model1.eval()

            output0 = model0(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                unmasked_patch_features=visual_embeds,
                visual_mask=visual_mask,
                visual_token_type_id=visual_token_type_id,
            )
            output0 = torch.sigmoid(output0)  # 属性
            output1 = model1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                unmasked_patch_features=visual_embeds,
                visual_mask=visual_mask,
                visual_token_type_id=visual_token_type_id,
            )
            output1 = torch.sigmoid(output1)  # 图文

            label_masks = label_masks.cpu()

            # TODO 修改路径
            with open('../../data/tmp_data/logits/nezhaB.txt', 'a', encoding='utf-8') as f:
                for i in np.arange(len(names)):  # batch里面的第i个人
                    name = names[i]
                    out1 = {"img_name": name}
                    out2 = {}
                    idx = np.where(label_masks[i, :] == 1)[0]
                    if opt.p == 0:
                        # 属性网络进行预测
                        for j in idx:
                            kn = label_list[j]  # 获取对应属性值
                            out2[kn] = int(output0[i, j].item() > 0.5)  # 属性值对应预测值

                        # 图文网络进行预测
                        out2[label_list[0]] = int(output1[i, 0].item() > 0.5)
                        # 修正
                        # if output1[i, 0] >= 0.8:
                        #     for key in out2.keys():
                        #         out2[key] = 1

                        out1.update({"match": out2})
                    else:
                        # 属性网络进行预测
                        for j in idx:
                            kn = label_list[j]  # 获取对应属性值
                            out2[kn] = output0[i, j].item()  # 属性值对应预测值
                        # 图文网络进行预测
                        out2[label_list[0]] = output1[i, 0].item()
                        out1.update({"match": out2})

                    js = json.dumps(out1, ensure_ascii=False)
                    f.write(js)
                    f.write('\n')

        print('Finish testing')


if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    main(opt)
