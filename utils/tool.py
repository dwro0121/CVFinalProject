import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image
from torch.autograd import Variable
from tqdm import tqdm

from utils.losses import focal_loss, l1_loss

number_of_colors = 80

colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(number_of_colors)]


def preprocess_image(image):
    # return np.float32(image/255.)
    mean = [0.3915, 0.4243, 0.4478]
    std = [0.2354, 0.2332, 0.2391]
    return ((np.float32(image) / 255.) - mean) / std


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
                                'wh_loss': 0.1 * wh_loss.item(),
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


def detect_image(net, test_loader, class_names, cuda):
    net.eval()
    print(len(test_loader))
    with tqdm(postfix=dict, mininterval=0.3) as pbar:
        for iteration, data in enumerate(test_loader):
            with torch.no_grad():
                img = detect_img(net, Image.fromarray(data[0].numpy()), class_names, cuda)
                import matplotlib.pyplot as plt
                plt.imshow(img)
                plt.show()


def detect_img(net, img, class_names, cuda):
    img_shape = np.array(np.shape(img)[0:2])
    image = np.array(img, dtype=np.float32)[:, :, ::-1]
    image = np.reshape(np.transpose(preprocess_image(image), (2, 0, 1)), [1, 3, 416, 416])
    with torch.no_grad():
        image = np.asarray(image)
        images = Variable(torch.from_numpy(image).type(torch.FloatTensor))
        if cuda:
            images = images.cuda()
        ret = net(images)
        hm, wh, offset = ret['hm'], ret['wh'], ret['offsets']
        detected_boxes = process_bbox(hm, wh, offset, 0.05, cuda, 50)
    detected_boxes = np.array(nms(detected_boxes, 0.05))
    if len(detected_boxes) <= 0:
        return img
    bbox = detected_boxes[0]
    if len(bbox) <= 0:
        return img

    batch_boxes, conf, label = bbox[:, :4], bbox[:, 4], bbox[:, 5]
    xmin, ymin, xmax, ymax = batch_boxes[:, 0], batch_boxes[:, 1], batch_boxes[:, 2], batch_boxes[:, 3]
    top_indices = [i for i, cf in enumerate(conf) if cf >= 0.05]
    top_conf = conf[top_indices]
    top_label_indices = label[top_indices].tolist()
    left, top, right, bottom = np.expand_dims(xmin[top_indices], -1), np.expand_dims(ymin[top_indices],
                                                                                     -1), np.expand_dims(
        xmax[top_indices], -1), np.expand_dims(ymax[top_indices], -1)
    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
    box_hw = np.concatenate((bottom - top, right - left), axis=-1)
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([img_shape, img_shape], axis=-1)

    font = ImageFont.truetype(font='../data/simhei.ttf', size=np.floor(3e-2 * np.shape(img)[1] + 0.5).astype('int32'))

    thickness = (np.shape(img)[0] + np.shape(img)[1]) // 416
    for i, c in enumerate(top_label_indices):
        predicted_class = class_names[int(c)]
        score = top_conf[i]

        top, left, bottom, right = boxes[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(img)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(img)[1], np.floor(right + 0.5).astype('int32'))
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[int(c)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[int(c)])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    return img


def pool(hm, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == hm).float()
    return hm * keep


def process_bbox(hm, wh, offset, threshold, cuda, topk=100):
    pred_hm = pool(hm)
    batch, channel, h, w = pred_hm.shape
    detected_box = []
    for b in range(batch):
        heat_map = pred_hm[b].permute(1, 2, 0).view([-1, channel])
        wh_map = wh[b].permute(1, 2, 0).view([-1, 2])
        offset_map = offset[b].permute(1, 2, 0).view([-1, 2])

        mesh_y, mesh_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))

        mesh_x, mesh_y = mesh_x.flatten().float(), mesh_y.flatten().float()
        if cuda:
            mesh_x = mesh_x.cuda()
            mesh_y = mesh_y.cuda()

        conf, pred = torch.max(heat_map, dim=-1)
        mask = conf > threshold

        wh_mask = wh_map[mask]
        offset_mask = offset_map[mask]
        if len(wh_mask) == 0:
            detected_box.append([])
            continue
        x_mask = torch.unsqueeze(mesh_x[mask] + offset_mask[..., 0], -1)
        y_mask = torch.unsqueeze(mesh_y[mask] + offset_mask[..., 0], -1)

        half_w, half_h = wh_mask[..., 0:1] / 2, wh_mask[..., 1:2] / 2
        bboxes = torch.cat([x_mask - half_w, y_mask - half_h, x_mask + half_w, y_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h
        detected = torch.cat([bboxes, torch.unsqueeze(conf[mask], -1), torch.unsqueeze(pred[mask], -1).float()], dim=-1)
        arg_sort = torch.argsort(detected[:, -2], descending=True)
        detected = detected[arg_sort]
        detected_box.append(detected.cpu().numpy()[:topk])
    return detected_box


def nms(detected_box, threshold):
    outputs = []
    for i in range(len(detected_box)):
        detections = detected_box[i]
        best_box = []
        if len(detections) == 0:
            detected_box.append(best_box)
            continue
        unique_class = np.unique(detections[:, -1])
        if len(unique_class) == 0:
            detected_box.append(best_box)
            continue
        for c in unique_class:
            cls_mask = detections[:, -1] == c

            detection = detections[cls_mask]
            scores = detection[:, 4]
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0] > 0:
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1], detection[1:])
                detection = detection[1:][ious < threshold]
        outputs.append(best_box)
    return outputs


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou


def process(hm, wh, offset, threshold=0.3, cuda=None, peaks_num=100):
    if cuda:
        hmax = nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1).cuda()
    else:
        hmax = nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
    keep = (hmax == hm).float()
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
    topk_hm, topk_id = torch.topk(hm.squeeze(0).view(c, -1), peaks_num, dim=1)
    mesh_w = mesh_w + offset[:, 0]
    mesh_h = mesh_h + offset[:, 1]
    half_w = wh[:, 0] / 2
    half_h = wh[:, 1] / 2
    bboxes = np.array([[[topk_hm[i, j].item(),
                         (mesh_w - half_w)[topk_id[i, j]].item() * 4,
                         (mesh_h - half_h)[topk_id[i, j]].item() * 4,
                         (mesh_w + half_w)[topk_id[i, j]].item() * 4,
                         (mesh_h + half_h)[topk_id[i, j]].item() * 4]
                        for j in range(peaks_num)] for i in range(c)])
    return bboxes
