import torch
import torch.nn.functional as F


def gaussian_kernel(Y):
    '''

    :return:
    '''


def focal_loss(output, label, alpha=2, beta=4):
    # label = gaussian_kernel(label)
    output = output.permute(0, 2, 3, 1)
    pos_inds = label.eq(1).float()
    neg_inds = label.lt(1).float()

    output = torch.clamp(output, 1e-12, 1 - 1e-12)

    pos_loss = pos_inds * torch.log(output) * torch.pow(1 - output, alpha)
    neg_loss = neg_inds * torch.log(1 - output) * torch.pow(output, alpha) * torch.pow(1 - label, beta)

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def l1_loss(output, label, mask):
    '''

    :param output:
    :param label:
    :param mask:
    :return:
    '''
    output = output.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(output * expand_mask, label * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
