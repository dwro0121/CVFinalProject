import math
from random import shuffle

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


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


def preprocess_image(image):
    mean = [0.22083542, 0.22083542, 0.22083542]
    std = [0.33068898, 0.3250564, 0.32701415]
    return ((np.float32(image) / 255.) - mean) / std


class Dataset(Dataset):
    def __init__(self, train_lines, input_size, num_classes):
        super(Dataset, self).__init__()

        self.train_lines = train_lines
        self.input_size = input_size
        self.output_size = (int(input_size[0] / 4), int(input_size[1] / 4))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        line = self.train_lines[index].split()
        img = Image.open(line[0])
        y = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes),
                            dtype=np.float32)
        batch_wh = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)

        if len(y) != 0:
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.input_size[1] * self.output_size[1]
            boxes[:, 1] = boxes[:, 1] / self.input_size[0] * self.output_size[0]
            boxes[:, 2] = boxes[:, 2] / self.input_size[1] * self.output_size[1]
            boxes[:, 3] = boxes[:, 3] / self.input_size[0] * self.output_size[0]

        for i in range(len(y)):
            bbox = boxes[i].copy()
            bbox = np.array(bbox)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size[1] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size[0] - 1)
            cls_id = int(y[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)

                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        img = np.array(img, dtype=np.float32)[:, :, ::-1]
        img = np.transpose(preprocess_image(img), (2, 0, 1))

        return img, batch_hm, batch_wh, batch_reg, batch_reg_mask


# DataLoader中collate_fn使用
def collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_whs = np.array(batch_whs)
    batch_regs = np.array(batch_regs)
    batch_reg_masks = np.array(batch_reg_masks)
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks
