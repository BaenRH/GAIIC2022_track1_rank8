# _*_ coding: utf-8 _*_
"""
Time:     2022/5/8 14:45
Author:   Cai Ruihan
File:     check_weight.py
"""
import torch
from model_nezha import NeZhaForJointLSTM
# finetune_dir = 'save_finetune/target_2/pytorch_model.bin'
# fea_dir = 'save_fea_clf/target_2/pytorch_model.bin'
#
# weight_fine = torch.load(finetune_dir)
# weight_fea = torch.load(fea_dir)
# print(weight_fine['bert.encoder.layer.0.attention.self.query.weight'])
# print(weight_fea['nezhabase.bert.encoder.layer.0.attention.self.query.weight'])
# print(weight_fine['text_embedding.weight'] == weight_fea['text_embedding.weight'])
#

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}