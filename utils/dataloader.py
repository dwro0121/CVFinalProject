import math
import os
from random import *

import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data.dataset import Dataset


def augmentor(img, box, img_size):
    img = np.array(img)
    box_list = []
    for data in box:
        box_list.append(BoundingBox(data[0], data[1], data[2], data[3]))
    bbs = BoundingBoxesOnImage(box_list, shape=img.shape)
    seq = iaa.Sequential([
        iaa.LinearContrast((0.75, 1.25), seed=randint(0, 1000)),
        iaa.Multiply((0.75, 1.25), seed=randint(0, 1000)),
        iaa.Fliplr(0.5, seed=randint(0, 1000)),
        iaa.Crop(percent=(0, 0.3), seed=randint(0, 1000))],
        seed=randint(0, 1000)
    )

    img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    for i in range(len(bbs_aug.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]
        box[i, :4] = [after.x1, after.y1, after.x2, after.y2]
    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, img_size[1] - 1)
    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, img_size[0] - 1)
    list_del = []
    for i in range(len(box)):
        if math.isclose(box[i, 0], box[i, 2]) or math.isclose(box[i, 1], box[i, 3]):
            list_del.append(i)
    list_del.sort(reverse=True)
    for i in range(len(list_del)):
        np.delete(box, i)
    return img, box


def preprocess_image(image):
    mean = [0.3792, 0.4117, 0.4419]
    std = [0.2385, 0.2373, 0.2451]
    return ((np.float32(image) / 255.) - mean) / std


class Dataset(Dataset):
    def __init__(self, lines, input_size, num_classes, augment=True, tvt='train'):
        super(Dataset, self).__init__()

        self.tvt = tvt
        self.lines = lines
        self.input_size = input_size
        self.output_size = (int(input_size[0] / 4), int(input_size[1] / 4))
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.lines)
        line = self.lines[index].split()
        img = Image.open(line[0])
        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        if self.augment:
            img, box = augmentor(img, box, self.input_size)

        batch_hm = np.zeros((self.output_size[0], self.output_size[1], self.num_classes),
                            dtype=np.float32)
        batch_wh = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg = np.zeros((self.output_size[0], self.output_size[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)

        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.input_size[1] * self.output_size[1]
            boxes[:, 1] = boxes[:, 1] / self.input_size[0] * self.output_size[0]
            boxes[:, 2] = boxes[:, 2] / self.input_size[1] * self.output_size[1]
            boxes[:, 3] = boxes[:, 3] / self.input_size[0] * self.output_size[0]

        for i in range(len(box)):
            bbox = boxes[i].copy()
            bbox = np.array(bbox)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size[1] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size[0] - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        if self.tvt == 'test':
            basename = os.path.basename(line[0])
            basename_no_ext = os.path.splitext(basename)[0]
            return np.array(img), basename_no_ext

        img = np.array(img, dtype=np.float32)[:, :, ::-1]
        img = np.transpose(preprocess_image(img), (2, 0, 1))
        return img, batch_hm, batch_wh, batch_reg, batch_reg_mask


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
