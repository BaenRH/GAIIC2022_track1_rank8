# _*_ coding: utf-8 _*_
"""
Time:     2022/3/24 20:59
Author:   Cai Ruihan
File:     build_optimizer.py
"""

from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW
import math
import re

def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']  # TODO：指哪的bias和哪里的LayerNorm

    bert_param_optimizer = list(model.bert.named_parameters()) + \
                           list(model.im_to_embedding.named_parameters()) + \
                           list(model.text_embedding.named_parameters())

    # other_param_optimizer = list(model.classifier.named_parameters())
    # other_param_optimizer = list(model.lstm.named_parameters()) + \
    #                         list(model.intent_classifier.named_parameters()) + \
    #                         list(model.slot_classifier.named_parameters())

    other_param_optimizer = []
    for name, para in model.named_parameters():
        # 对于需要更新的参数：
        if para.requires_grad:
            if "bert" not in name and 'embedding' not in name:
                other_param_optimizer += [(name, para)]

    optimizer_grouped_parameters = [
        # 除了偏差项和归一化权重，对bert其他参数进行衰减
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
        # 除了偏差项和归一化权重，对其他网络参数进行衰减
        {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, 'lr': args.other_lr},  # 应该是单独的学习率
        #
        {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, )

    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    return optimizer, scheduler


def build_optimizer2(args, model, train_steps, frozen_num=10):
    # TODO:frozen_num  还没实现哈

    for i, j in model.named_parameters():
        if bool(re.search(r'bert.encoder.layer.[0-9]\.', i)):
            j.requires_grad=False
        if 'embedding' in i:
            j.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.lr, )

    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    return optimizer, scheduler


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))
