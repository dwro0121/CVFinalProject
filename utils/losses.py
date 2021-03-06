import torch
import torch.nn.functional as F


def focal_loss(output, label, alpha=2, beta=4):
    loss = .0
    output = output.permute(0, 2, 3, 1)
    pos_inds = label.eq(1).float()
    neg_inds = label.lt(1).float()

    output = torch.clamp(output, 1e-12)

    pos_loss = torch.log(output) * torch.pow(1 - output, alpha) * pos_inds
    neg_loss = torch.log(1 - output) * torch.pow(output, alpha) * torch.pow(1 - label, beta) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss += -neg_loss
    else:
        loss += -(pos_loss + neg_loss) / num_pos
    return loss


def l1_loss(output, label, mask):
    output = output.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(output * expand_mask, label * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-12)
    return loss
