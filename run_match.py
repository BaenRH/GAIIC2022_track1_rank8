# _*_ coding: utf-8 _*_
import os
import argparse
import json
import time
from tqdm import tqdm
from datasets import FinetuneDateset, process_data, clear
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    AdamW,
    set_seed,
)
from model_nezha import NezhaFinetune, NeZhaForJointLSTM, NeZhawithAttn, NeZhaFuse
from helper.FGM import FGM
from helper.build_optimizer import build_optimizer


def main(opt):
    # Set seed before initializing model.
    global coarse_unmatch_labels, coarse_unmatch_key_attrs
    set_seed(opt.seed)

    # redefine dir
    if opt.debug is True and opt.train_env == 0:  # 线下debug
        print('-----本地调试-----')
        coarse_unmatch_path = '../sample_v2/coarse_unmatch_data.json'
        opt.data_root = '../sample_v2'
    elif opt.debug is False and opt.train_env == 0:
        print('-----本地训练-----')
        opt.data_root = '../../fsf_v1/datasets/train'
        coarse_unmatch_path = '../data/train/coarse_unmatch_data.json'  # coarse中图文不匹配的数据 TODO: 注意改路径
    else:
        print('-----线上训练-----')
        # TODO：改地址
        # data_root = os.path.join('../input', os.path.split(opt.data_root)[-1])
        coarse_unmatch_path = 'coarse_unmatch_data.json'

    # 加载所有需要匹配的类别
    label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
    # 标签转化为对应id
    label2id = {key: i for i, key in enumerate(label_list)}
    print('匹配任务:')
    print(label2id)

    # 获取属性的所有值
    with open(os.path.join(opt.data_root, 'attr_to_attrvals.json'), 'r', encoding='utf-8') as f:
        key_attr_values = json.loads(f.read())

    """
    第三版，将coarse data转成fine data，与提供的fine data一起训练
    """

    # 文件路径
    coarse_to_fine_train_path = os.path.join(opt.data_root, 'coarse_to_fine_data.json')
    fine_train_path = os.path.join(opt.data_root, 'fine_data.json')

    print('开始加载数据')
    t1 = time.time()

    # 获取 fine data
    with open(fine_train_path, 'r') as f:
        fine_data = json.loads(f.read())
    fine_texts, fine_img_features, fine_labels, fine_label_masks, fine_key_attrs = \
        fine_data['texts'], fine_data['img_features'], fine_data['labels'], fine_data['label_masks'], fine_data[
            'key_attrs']
    # clear
    fine_texts = list(map(clear, fine_texts))
    print('fine data 数据量:', len(fine_texts))

    # 获取 coarse2fine data
    if opt.coarse2fine == 1:
        with open(coarse_to_fine_train_path, 'r') as f:
            coarse_to_fine_data = json.loads(f.read())
        coarse_to_fine_texts, coarse_to_fine_img_features, coarse_to_fine_labels, coarse_to_fine_label_masks, coarse_to_fine_key_attrs = \
            coarse_to_fine_data['texts'], coarse_to_fine_data['img_features'], coarse_to_fine_data['labels'], \
            coarse_to_fine_data['label_masks'], coarse_to_fine_data['key_attrs']
        # clear
        coarse_to_fine_texts = list(map(clear, coarse_to_fine_texts))
        print('coarse_to_fine data 数据量:', len(coarse_to_fine_texts))

    # 获取 unmatch data
    with open(coarse_unmatch_path, 'r') as f:
        coarse_unmatch_data = json.loads(f.read())
    coarse_unmatch_texts, coarse_unmatch_img_features, coarse_unmatch_labels, coarse_unmatch_label_masks, coarse_unmatch_key_attrs = \
        coarse_unmatch_data['texts'], coarse_unmatch_data['img_features'], coarse_unmatch_data['labels'], \
        coarse_unmatch_data['label_masks'], coarse_unmatch_data['key_attrs']
    # clear
    coarse_unmatch_texts = list(map(clear, coarse_unmatch_texts))
    print('coarse umatch 数量：', len(coarse_unmatch_texts))

    # # 获取fake data
    # with open('../data_process/fake_data.json', 'r', ) as f:
    #     fake_data = json.loads(f.read())
    #
    # fake_texts, fake_img_features, fake_labels, fake_label_masks, fake_key_attrs = \
    #     fake_data['texts'], fake_data['img_features'], fake_data['labels'], \
    #     fake_data['label_masks'], fake_data['key_attrs']
    # print('fake data数量：', len(fake_texts))

    t2 = time.time()
    print('读取数据花费的时间:', t2 - t1)

    """
    只使用fine data的数据作为线下验证
    """
    fine_data_texts = np.array(fine_texts)
    fine_data_img_features = np.array(fine_img_features)
    fine_data_labels = np.array(fine_labels)
    fine_data_label_masks = np.array(fine_label_masks)
    fine_data_key_attrs = np.array(fine_key_attrs)

    assert 0 <= opt.test_size < 1

    # 如果test_size为0，即全部数据一起训练
    if opt.test_size == 0:
        _, test_idxs = train_test_split(range(len(fine_data_texts)), test_size=0.2)
        train_idxs = [i for i in range(len(fine_data_texts))]
    else:
        # 划分训练集和测试集
        train_idxs, test_idxs = train_test_split(range(len(fine_data_texts)), test_size=opt.test_size)

    train_fine_texts = fine_data_texts[train_idxs]
    train_fine_img_features = fine_data_img_features[train_idxs]
    train_fine_labels = fine_data_labels[train_idxs]
    train_fine_label_masks = fine_data_label_masks[train_idxs]
    train_fine_key_attrs = fine_data_key_attrs[train_idxs]

    if opt.target == 1:
        # 添加 unmatch 的数据，只用于图文训练
        train_texts = np.concatenate((train_fine_texts, np.array(coarse_to_fine_texts), np.array(coarse_unmatch_texts)))
        train_img_features = np.concatenate(
            (train_fine_img_features, np.array(coarse_to_fine_img_features), np.array(coarse_unmatch_img_features)))
        train_labels = np.concatenate(
            (train_fine_labels, np.array(coarse_to_fine_labels), np.array(coarse_unmatch_labels)))
        train_label_masks = np.concatenate(
            (train_fine_label_masks, np.array(coarse_to_fine_label_masks), np.array(coarse_unmatch_label_masks)))
        train_key_attrs = np.concatenate(
            (train_fine_key_attrs, np.array(coarse_to_fine_key_attrs), np.array(coarse_unmatch_key_attrs)))
    elif opt.target == 2 and opt.mask == 1:  # 一起训练，且使用掩码
        # 添加 unmatch 的数据，改mask
        train_texts = np.concatenate((train_fine_texts, np.array(coarse_to_fine_texts), np.array(coarse_unmatch_texts)))
        train_img_features = np.concatenate(
            (train_fine_img_features, np.array(coarse_to_fine_img_features), np.array(coarse_unmatch_img_features)))
        train_labels = np.concatenate(
            (train_fine_labels, np.array(coarse_to_fine_labels), np.array(coarse_unmatch_labels)))
        # 只对train data操作
        train_label_masks = np.concatenate(
            (np.ones_like(train_fine_label_masks), np.ones_like(np.array(coarse_to_fine_label_masks)),
             np.array(coarse_unmatch_label_masks)))
        train_key_attrs = np.concatenate(
            (train_fine_key_attrs, np.array(coarse_to_fine_key_attrs), np.array(coarse_unmatch_key_attrs)))
    else:
        # 一起训练，不使用掩码  或  只训练属性
        # 是否使用coarse to fine的数据
        train_texts = np.concatenate((train_fine_texts, np.array(coarse_to_fine_texts)))
        train_img_features = np.concatenate((train_fine_img_features, np.array(coarse_to_fine_img_features)))
        train_labels = np.concatenate((train_fine_labels, np.array(coarse_to_fine_labels)))
        train_label_masks = np.concatenate((train_fine_label_masks, np.array(coarse_to_fine_label_masks)))
        train_key_attrs = np.concatenate((train_fine_key_attrs, np.array(coarse_to_fine_key_attrs)))

    test_texts = fine_data_texts[test_idxs]
    test_img_features = fine_data_img_features[test_idxs]
    test_labels = fine_data_labels[test_idxs]
    test_label_masks = fine_data_label_masks[test_idxs]
    test_key_attrs = fine_data_key_attrs[test_idxs]

    print('训练集规模:', len(train_texts))
    print('验证集规模:', len(test_texts))

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=opt.pretrain_dir)

    # if opt.target == 1 or opt.target == 2:
    #     p1 = 0.4  # 目前0.4效果比0.537好一点，0.537使得正负样本比较均衡
    # else:
    #     p1 = 0.3

    # 创建数据集
    train_dataset = FinetuneDateset(
        tokenizer,
        train_texts,
        train_labels,
        train_img_features,
        train_label_masks,
        train_key_attrs,
        key_attr_values,
        label2id,
    )

    val_dataset = FinetuneDateset(
        tokenizer,
        test_texts,
        test_labels,
        test_img_features,
        test_label_masks,
        test_key_attrs,
        key_attr_values,
        label2id,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
    )

    # 加载预训练好的权重
    if opt.model == 0:
        print('use NezhaFinetune')
        model = NezhaFinetune.from_pretrained(opt.pretrain_dir)
    elif opt.model == 1:
        print('use NeZhaForJointLSTM')
        model = NeZhaForJointLSTM.from_pretrained(opt.pretrain_dir)
    elif opt.model == 2:
        print('use NeZhawithAttn')
        model = NeZhawithAttn.from_pretrained(opt.pretrain_dir)
    elif opt.model == 3:
        print('use NezhaFuse')
        model = NeZhaFuse.from_pretrained(opt.pretrain_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    # TODO: 损失函数  可改
    loss_fct = nn.BCELoss()
    # 用于训练
    loss_fct_train = nn.BCEWithLogitsLoss()

    # 优化器
    if opt.schedule:
        total_steps = opt.epochs * len(train_dataloader)  # 总的训练batch数
        optim, lr_scheduler = build_optimizer(opt, model, total_steps)
    else:
        optim = AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    # 对抗训练
    fgm = FGM(model)

    # 保存每一次epoch的线下验证分数
    epoch_score = {}
    best_score = 0

    pbar = tqdm(total=opt.epochs)
    for epoch in range(opt.epochs):
        for batch in train_dataloader:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            visual_embeds = batch['visual_embeds'].to(device)
            visual_attention_mask = batch['visual_attention_mask'].to(device)
            visual_token_type_ids = batch['visual_token_type_ids'].to(device)

            labels = batch['labels'].to(device)
            label_masks = batch['label_masks'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            unmasked_patch_features=visual_embeds,
                            visual_mask=visual_attention_mask,
                            visual_token_type_id=visual_token_type_ids,
                            )

            # logits = torch.sigmoid(outputs)
            if opt.mask == 1:
                logits = outputs * label_masks
            else:
                logits = outputs
            """
            第一版对全部logits做掩码操作
            """
            # # 对输入进行掩码操作，这样不会对数据没有的query任务进行梯度更新
            # logits = logits * label_masks
            # loss = loss_fct(logits, labels)

            """
            第二版只对attr的logits做掩码操作
            第三版，取消掩码操作，好像没啥用
            """
            img_text_logits = logits[:, 0:1]
            attr_logits = logits[:, 1:]
            # 对attr的logit做掩码操作
            # 取消掩码操作
            # attr_logits = attr_logits * label_masks[:, 1:]
            img_text_labels = labels[:, 0:1]
            attr_labels = labels[:, 1:]

            # attr_mask = attr_labels > -1
            # mask_num = attr_mask.sum()

            img_text_loss = loss_fct_train(img_text_logits, img_text_labels)
            attr_loss = loss_fct_train(attr_logits, attr_labels)  # TODO check
            loss = img_text_loss + attr_loss

            if opt.target == 1:
                img_text_loss.backward()  # img
            elif opt.target == 0:
                attr_loss.backward()  # attr
            else:
                loss.backward()  # img、attr

            # -----------------------梯度截断-----------------------
            if opt.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # -----------------------对抗训练-----------------------
            if opt.attack:
                fgm.attack(epsilon=0.3, emb_name='text_embedding.weight')
                outputs_adv = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    unmasked_patch_features=visual_embeds,
                                    visual_mask=visual_attention_mask,
                                    visual_token_type_id=visual_token_type_ids,
                                    )
                # logits_adv = torch.sigmoid(outputs_adv)
                if opt.mask == 1:
                    logits_adv = outputs_adv * label_masks
                else:
                    logits_adv = outputs_adv
                img_text_logits_adv = logits_adv[:, 0:1]
                attr_logits_adv = logits_adv[:, 1:]
                img_text_loss_adv = loss_fct_train(img_text_logits_adv, img_text_labels)
                attr_loss_adv = loss_fct_train(attr_logits_adv, attr_labels)
                loss_adv = img_text_loss_adv + attr_loss_adv
                if opt.target == 1:
                    img_text_loss_adv.backward()
                elif opt.target == 0:
                    attr_loss_adv.backward()
                else:
                    loss_adv.backward()

                fgm.restore(emb_name='text_embedding.weight')
            # -----------------------------------------------------
            optim.step()

            if opt.schedule:
                lr_scheduler.step()

            pbar.set_postfix(loss=loss.item(),
                             img_text_loss=img_text_loss.item(),
                             attr_loss=attr_loss.item(),
                             lr=optim.state_dict()['param_groups'][0]['lr'])

        # 评估
        N_img_text = 0
        M_img_text = 0
        N_attr = 0
        M_attr = 0
        eval_losses = []
        eval_img_text_losses = []
        eval_attr_losses = []
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                visual_embeds = batch['visual_embeds'].to(device)
                visual_attention_mask = batch['visual_attention_mask'].to(device)
                visual_token_type_ids = batch['visual_token_type_ids'].to(device)

                labels = batch['labels'].to(device)
                label_masks = batch['label_masks'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                unmasked_patch_features=visual_embeds,
                                visual_mask=visual_attention_mask,
                                visual_token_type_id=visual_token_type_ids,
                                )

                logits = torch.sigmoid(outputs)

                # 验证集损失函数
                # loss = loss_fct(logits*label_masks, labels)

                img_text_logits = logits[:, 0]
                attr_logits = logits[:, 1:]
                # 对attr的logit做掩码操作
                # attr_logits = attr_logits * label_masks[:, 1:]
                img_text_labels = labels[:, 0]
                attr_labels = labels[:, 1:]

                img_text_loss = loss_fct(img_text_logits, img_text_labels)
                attr_loss = loss_fct(attr_logits, attr_labels)
                loss = img_text_loss + attr_loss

                eval_losses.append(loss)
                eval_img_text_losses.append(img_text_loss)
                eval_attr_losses.append(attr_loss)

                # 统计图文匹配数量
                # pred_img_text_logit = logits[:, 0]
                pred_img_text = torch.zeros_like(img_text_logits)
                pred_img_text[img_text_logits >= 0.5] = 1
                true_img_text = labels[:, 0]
                N_img_text += torch.sum(pred_img_text == true_img_text)
                M_img_text += torch.sum(torch.ones_like(true_img_text))

                # 统计属性匹配数量
                # pred_attr_logit = logits[:, 1:]
                pred_attr = torch.zeros_like(attr_logits)
                pred_attr[attr_logits >= 0.5] = 1
                pred_attr = pred_attr[label_masks[:, 1:] == 1]
                true_attr = labels[:, 1:]
                true_attr = true_attr[label_masks[:, 1:] == 1]
                N_attr += torch.sum(pred_attr == true_attr)
                M_attr += torch.sum(torch.ones_like(true_attr))

            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_img_text_loss = sum(eval_img_text_losses) / len(eval_img_text_losses)
            eval_attr_loss = sum(eval_attr_losses) / len(eval_attr_losses)
            img_text_scores = 0.5 * N_img_text / M_img_text
            attr_scores = 0.5 * N_attr / M_attr
            total_scores = img_text_scores + attr_scores

        val_str = f"[{epoch + 1:04d}/{opt.epochs:04d}] \t val loss:{eval_loss.item():.4f} \t " + \
                  f"val img_text_loss:{eval_img_text_loss.item():.4f} \t " + \
                  f"val eval_attr_loss:{eval_attr_loss.item():.4f} \t score:{total_scores.item():.4f} \t " + \
                  f"img_text_scores:{img_text_scores.item():.4f} \t attr_scores:{attr_scores.item():.4f}"
        tqdm.write(val_str)

        epoch_score[f'epoch_{epoch + 1}'] = total_scores.item()

        model.train()

        # 保存模型
        if opt.target == 1:
            # 图文
            total_scores = img_text_scores
        elif opt.target == 0:
            # attr
            total_scores = attr_scores
        else:
            # 图文 + attr
            total_scores = total_scores

        if total_scores.item() > best_score:
            best_score = total_scores.item()

            # 它包装在PyTorch DistributedDataParallel或DataParallel中
            model_to_save = model.module if hasattr(model, 'module') else model
            # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
            # os.makedirs(opt.output_dir, exist_ok=True)
            os.makedirs(os.path.join(opt.output_dir, f'target_{opt.target}_{opt.model}'), exist_ok=True)
            output_model_file = os.path.join(opt.output_dir, f'target_{opt.target}_{opt.model}', 'pytorch_model.bin')
            output_config_file = os.path.join(opt.output_dir, f'target_{opt.target}_{opt.model}', 'config.json')
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(os.path.join(opt.output_dir, f'target_{opt.target}_{opt.model}'))

        pbar.update(1)

    print('best score:', best_score)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='随机种子')

    #
    parser.add_argument('--train_env', type=int, default=1, help='线上训练：1；线下：0。用于修改文件路径')
    parser.add_argument('--debug', type=str, default=False, help='debug？')
    # dir
    parser.add_argument('--pretrain_dir', type=str, default='../../../temp/save_pretrain', help='预训练模型的目录')
    parser.add_argument('--data_root', type=str, default='../../../input/train_json6945', help='json数据目录')
    # parser.add_argument('--data_root', type=str, default='../sample_v2', help='数据目录')

    # param
    parser.add_argument('--output_dir', type=str, default='save_finetune', help='保存路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=400, help='batch_size')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')
    parser.add_argument('--mask', type=int, default=1, help='是否使用掩码')
    parser.add_argument('--other_lr', type=float, default=4e-5, help='other_lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup_ratio')

    # 训练目标
    parser.add_argument('--target', type=int, default=2, help='训练图文还是属性值。 1:图文；0：属性；2：全部')
    parser.add_argument('--coarse2fine', type=int, default=1, help='是否使用coarse to fine。 0：不用；1使用；')
    parser.add_argument('--model', type=int, default=1, help='选择模型：'
                                                             '0 NezhaFinetune,'
                                                             '1 NeZhaForJointLSTM'
                                                             '2 NeZhawithAttn'
                                                             '3 NezhaFuse')

    # trick
    parser.add_argument('--attack', type=int, default=0, help='是否对抗训练')
    parser.add_argument('--schedule', type=int, default=1, help='是否使用schedule')
    parser.add_argument('--clip', type=int, default=1, help='是否梯度截断')

    parser.add_argument('--save_epoch', type=int, default=1, help='模型保存的频次')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader的num_workers参数')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集划分')

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
