import os
import argparse
import torch
import re
from tqdm import tqdm
import numpy as np
import math
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from transformers import set_seed, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import *
from model_nezha import NezhaPretrain
from AdaptiveLoss import adaptive_loss
from helper.build_optimizer import WarmupCosineSchedule
from transformers import AdamW
import jieba


def train(opt, model, device, optimizer, scheduler, train_loader, test_loader):
    # 迭代次数
    iter_count = 0
    epoch_size = math.ceil(len(train_loader.dataset) / opt.batch_size)
    best_loss = float("inf")

    test_match_loss_plot = []
    train_loss_plot = []
    test_loss_plot = []

    for epoch in range(opt.epochs):

        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{opt.epochs}', postfix=dict,
                  mininterval=0.3) as pbar:
            # 训练模式
            model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device=device)  # batchsize*text_length
                token_type_ids = batch['token_type_ids'].to(device=device)  # batchsize*text_length
                attention_mask = batch['attention_mask'].to(device=device)
                visual_embeds = batch['visual_embeds'].to(device=device)
                visual_mask = batch['visual_attention_mask'].to(device=device)
                visual_token_type_id = batch['visual_token_type_ids'].to(device=device)
                labels = batch['labels'].to(device=device)
                is_pared = batch['sentence_image_labels'].to(device=device)

                optimizer.zero_grad()
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               token_type_ids=token_type_ids,
                               unmasked_patch_features=visual_embeds,
                               visual_mask=visual_mask,
                               visual_token_type_id=visual_token_type_id,
                               is_paired=is_pared,
                               device=device
                               )

                # loss = output['masked_lm_loss'] + output['alignment_loss']
                loss = adaptive_loss(output, device)  # TODO:Adaptive Loss  任务权重设置，2个任务该如何修改？
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix(**{'total_loss': total_loss / (batch_idx + 1),
                                    'lr': optimizer.state_dict()['param_groups'][0]['lr'], })
                pbar.update(1)
            train_loss_plot.append(total_loss / (batch_idx + 1))

        # 验证模式
        test_nums = math.ceil(len(test_loader.dataset) / opt.batch_size)
        with tqdm(total=test_nums, desc=f'Epoch {epoch + 1}/{opt.epochs}', postfix=dict,
                  mininterval=0.3) as pbar:

            total_loss_val = 0.0
            total_match_loss_val = 0.0
            total_num = 0
            total_right_num = 0

            with torch.no_grad():
                model.eval()
                for batch_idx, batch in enumerate(test_loader):
                    input_ids = batch['input_ids'].to(device=device)  # batchsize*text_length
                    token_type_ids = batch['token_type_ids'].to(device=device)  # batchsize*text_length
                    attention_mask = batch['attention_mask'].to(device=device)
                    visual_embeds = batch['visual_embeds'].to(device=device)
                    visual_mask = batch['visual_attention_mask'].to(device=device)
                    visual_token_type_id = batch['visual_token_type_ids'].to(device=device)
                    labels = batch['labels'].to(device=device)
                    is_pared = batch['sentence_image_labels'].to(device=device)

                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels,
                                   token_type_ids=token_type_ids,
                                   unmasked_patch_features=visual_embeds,
                                   visual_mask=visual_mask,
                                   visual_token_type_id=visual_token_type_id,
                                   is_paired=is_pared,
                                   device=device
                                   )
                    # loss = adaptive_loss(output)
                    loss = output['masked_lm_loss'] + output['alignment_loss']

                    total_match_loss_val += output['alignment_loss'].item()
                    total_loss_val += loss.item()

                    total_right_num += output['right_match']
                    total_num += len(is_pared)

                    pbar.set_postfix(**{'total_loss_val': total_loss_val / (batch_idx + 1),
                                        'total_match_loss_val': total_match_loss_val / (batch_idx + 1),
                                        'lr': optimizer.state_dict()['param_groups'][0]['lr'], })
                    pbar.update(1)

                test_loss_plot.append(total_loss_val / (batch_idx + 1))
                test_match_loss_plot.append(total_match_loss_val / (batch_idx + 1))

            print('match_score:', total_right_num / total_num)
            if best_loss > total_loss_val:
                best_loss = total_loss_val

                # 它包装在PyTorch DistributedDataParallel或DataParallel中
                model_to_save = model.module if hasattr(model, 'module') else model
                # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
                os.makedirs(opt.output_dir, exist_ok=True)
                output_model_file = os.path.join(opt.output_dir, 'pytorch_model.bin')
                output_config_file = os.path.join(opt.output_dir, 'config.json')
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(opt.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--debug', type=str, default=False, help='debug？')
    parser.add_argument('--train_env', type=int, default= 1, help='线上训练：1；线下：0。用于修改文件路径')

    # dir
    parser.add_argument('--seed', type=int, default=2022, help='随机种子')
    parser.add_argument('--tokenizer_path', type=str, default='new-base', help='tokenizer的路径')
    # TODO check
    parser.add_argument('--data_root', type=str, default='../input/track1_contest_4362/train/train/', help='数据目录')
    # parser.add_argument('--data_root', type=str, default='../../dataset/train', help='数据目录')
    # parser.add_argument('--data_root', type=str, default='../sample_v2', help='数据目录')  # 小数据
    parser.add_argument('--output_dir', type=str, default='../../../temp/save_pretrain', help='保存路径')   # TODO check

    # param
    parser.add_argument('--epochs', type=int, default=50, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=300, help='batch_size')
    parser.add_argument('--lr', type=float, default=5e-5, help='lr')
    parser.add_argument('--other_lr', type=float, default=4e-5, help='other_lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup_ratio')

    opt = parser.parse_args()
    print(opt)
    set_seed(opt.seed)

    # redefine dir
    if opt.debug is True and opt.train_env == 0:
        print('-----本地调试-----')
        opt.data_root = '../sample_v2'
    elif opt.debug is False and opt.train_env == 0:
        opt.data_root = '../../dataset/train'
        print('-----本地训练-----')
    else:  # 线上
        # TODO check
        data_root = os.path.join('../..',opt.data_root)
        print('data root:',data_root)
        print('-----线上训练-----')

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=opt.tokenizer_path)

    # 文件
    # coarse_train_path = os.path.join(opt.data_root, 'train_coarse.txt')
    # fine_train_path = os.path.join(opt.data_root, 'train_fine.txt')
    # 5.20 改
    coarse_train_path = os.path.join(data_root, 'train_coarse.txt')
    fine_train_path = os.path.join(data_root, 'train_fine.txt')

    # 加载所有需要匹配的类别
    label_list = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
    # 标签转化为对应id
    label2id = {key: i for i, key in enumerate(label_list)}

    # 获取key-attrs所有值
    with open('attr_to_attrvals.json', 'r', encoding='utf-8') as f:  # TODO
        key_attr_values = json.loads(f.read())

    # load  data
    fine_img_names, fine_texts, fine_img_features, fine_labels, fine_label_masks, fine_key_attrs = \
        process_data(fine_train_path, label2id)
    print('Finish loading fine data')
    coarse_img_names, coarse_texts, coarse_img_features, coarse_labels, coarse_label_masks, coarse_key_attrs = \
        process_data(coarse_train_path, label2id)
    print('Finish loading coarse data')

    # 拼接数据
    # data_texts = np.array(fine_texts+coarse_texts)
    data_img_features = np.array(fine_img_features + coarse_img_features)
    data_labels = np.array(fine_labels + coarse_labels)
    data_label_masks = np.array(fine_label_masks + coarse_label_masks)
    data_key_attrs = np.array(fine_key_attrs + coarse_key_attrs)

    # title处理
    data_texts = np.array(list(map(clear, list(fine_texts) + list(coarse_texts))))  # test

    # 划分训练集和测试集
    train_idxs, test_idxs = train_test_split(range(len(data_texts)), test_size=.2)

    train_texts = data_texts[train_idxs]
    train_img_features = data_img_features[train_idxs]
    train_labels = data_labels[train_idxs]
    train_label_masks = data_label_masks[train_idxs]
    train_key_attrs = data_key_attrs[train_idxs]

    test_texts = data_texts[test_idxs]
    test_img_features = data_img_features[test_idxs]
    test_labels = data_labels[test_idxs]
    test_label_mask = data_label_masks[test_idxs]
    test_key_attrs = data_key_attrs[test_idxs]

    #
    train_dataset = PretrainDataset(
        tokenizer,
        train_texts,
        train_img_features,
        train_labels,
        train_key_attrs,
        key_attr_values,
    )
    val_dataset = PretrainDataset(
        tokenizer,
        test_texts,
        test_img_features,
        test_labels,
        test_key_attrs,
        key_attr_values,
    )

    # data_collator，可以让输入带掩码的数据
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, )
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=opt.batch_size
    )
    val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=opt.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = BertConfig(
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        max_relative_position=64,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=2,
        vocab_size=1543,
        use_relative_position=True
    )
    model = NezhaPretrain(config=config)
    model.to(device=device)

    total_steps = opt.epochs * len(train_dataloader)  # 总的训练batch数
    optimizer = AdamW(model.parameters(), lr=opt.lr, betas=(0.95, 0.999), weight_decay=opt.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=total_steps * opt.warmup_ratio, t_total=total_steps)

    train(opt=opt, model=model, device=device, optimizer=optimizer, scheduler=scheduler,
          train_loader=train_dataloader,
          test_loader=val_dataloader)
