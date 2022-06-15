
import torch


def adaptive_loss(outputs, device):
    masked_lm_loss = outputs['masked_lm_loss']
    masked_patch_loss = torch.tensor(0.0).to(device=device)   # .cuda()  # TODO:adaptive_loss对于2个任务该如何修改
    alignment_loss = outputs['alignment_loss']

    G = torch.stack([masked_lm_loss, alignment_loss, masked_patch_loss])  # [3]
    w0 = 1.0
    w1 = 1.0
    w2 = 1.0
    isAdaptive = True
    if isAdaptive:
        logits = torch.nn.Softmax(dim=0)(G)
        nG = logits * logits
        alpha = 1.0
        K = 3.0
        denominator = (alpha * K - nG[0]) * (alpha * K - nG[1]) + (alpha * K - nG[1]) * (alpha * K - nG[2]) + (
                    alpha * K - nG[2]) * (alpha * K - nG[0])
        w0 = (alpha * K - nG[1]) * (alpha * K - nG[2]) / denominator
        w1 = (alpha * K - nG[2]) * (alpha * K - nG[0]) / denominator
        w2 = (alpha * K - nG[0]) * (alpha * K - nG[1]) / denominator

    adaptive_loss = w0 * masked_lm_loss + w1 * alignment_loss + w2 * masked_patch_loss
    return adaptive_loss
