import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from utils.losses import focal_loss, l1_loss

list_color = [(255,0,0), (0,255,0), (0,0,255), (0,0,0), (217,65,197)]

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = eval(f.readlines()[0])
    return class_names


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(net, epoch, epoch_size, train_loader, Epoch_Num, cuda, optimizer):
    total_loss = 0
    start_time = time.time()
    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch_Num}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_size:
                break

            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            optimizer.zero_grad()

            ret = net(batch_images)
            hm, wh, offset = ret['hm'], ret['wh'], ret['offsets']
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = l1_loss(offset, batch_regs, batch_reg_masks)

            loss = c_loss + 0.1 * wh_loss + off_loss

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'c_loss': c_loss.item(),
                                'wh_loss': 0.1*wh_loss.item(),
                                'off_loss': off_loss.item(),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)
            start_time = time.time()
    return total_loss / (epoch_size + 1)


def val_one_epoch(net, epoch, epoch_size, val_loader, Epoch_Num, cuda):
    net.eval()
    val_loss = 0
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch_Num}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= epoch_size:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                ret = net(batch_images)
                hm, wh, offset = ret['hm'], ret['wh'], ret['offsets']

                c_loss = focal_loss(hm, batch_hms)
                wh_loss = l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = l1_loss(offset, batch_regs, batch_reg_masks)
                loss = c_loss + 0.1 * wh_loss + off_loss
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    return val_loss / (epoch_size + 1)


def detect_image(net, test_loader, cuda):
    net.eval
    print(len(test_loader))
    with tqdm(postfix=dict, mininterval=0.3) as pbar:
        for iteration, data in enumerate(test_loader):
            with torch.no_grad():
                if cuda:
                    img = Variable(data[0].type(torch.FloatTensor)).cuda()
                else:
                    img = Variable(data[0].type(torch.FloatTensor))

                ret = net(img)
                hm, wh, offset = ret['hm'], ret['wh'], ret['offsets']
                bboxes = process(hm, wh, offset,0.1, cuda)
                img = np.array(img.cpu().detach()).squeeze(0)
                c, n = np.shape(bboxes)[0],np.shape(bboxes)[1]
                img = np.transpose(img,(1,2,0))
                for i in range(c):
                    list2 = []
                    for j in range(n):
                        list2.append(BoundingBox(int(bboxes[i,j,1]),int(bboxes[i,j,2]),int(bboxes[i,j,3]),int(bboxes[i,j,4])))
                    bbs = BoundingBoxesOnImage(list2, shape=np.shape(img))
                    img = bbs.draw_on_image(img, color=list_color[i],size=2)
                import matplotlib.pyplot as plt

                plt.imshow(img)
                plt.show()


def process(hm, wh, offset,threshold=0.3, cuda=None, peaks_num=20):
    if cuda:
        hmax = nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1).cuda()
    else:
        hmax = nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
    keep = (hmax == hm).float()
    keep[:, :, 0, :] = 0
    keep[:, :, :, 0] = 0
    keep[:, :, -1, :] = 0
    keep[:, :, :, -1] = 0
    mask = (hm > threshold).float()
    hm = hm * keep * mask
    b, c, h, w = hm.shape

    wh = wh.squeeze(0).permute(1, 2, 0).view(-1, 2)
    offset = offset.squeeze(0).permute(1, 2, 0).view(-1, 2)
    mesh_w, mesh_h = torch.meshgrid(torch.arange(h), torch.arange(w))
    if cuda:
        mesh_w, mesh_h = torch.flatten(mesh_w).float().cuda(), torch.flatten(mesh_h).float().cuda()
    else:
        mesh_w, mesh_h = torch.flatten(mesh_w).float(), torch.flatten(mesh_h).float()
    topk_hm, topk_id = torch.topk(hm[:].view(c, -1), peaks_num, dim=1)
    mesh_w = mesh_w + offset[:, 0]
    mesh_h = mesh_h + offset[:, 1]
    half_w = wh[:, 0] / 2
    half_h = wh[:, 1] / 2
    bboxes = np.array([[[topk_hm[i, j].item(),
                (mesh_w - half_w)[topk_id[i,j]].item()*4,
                (mesh_h - half_h)[topk_id[i,j]].item()*4,
                (mesh_w + half_w)[topk_id[i,j]].item()*4,
                (mesh_h + half_w)[topk_id[i,j]].item()*4]
               for j in range(peaks_num)] for i in range(c)])
    return bboxes


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
