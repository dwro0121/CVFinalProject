import glob
import os
import xml.etree.ElementTree as ET
from os import getcwd

import imgaug.augmenters as iaa
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from sklearn.model_selection import train_test_split
import numpy as np

dirs = ['../data/VOC2007/JPEGImages/']
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append([filename])
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append([filename])

    return image_list


def convert_annotation(file, dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml', encoding='UTF8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    img = Image.open(image_path)
    img = np.array(img)
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    list1 = []
    list2 = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (xmlbox.find('xmin').text, xmlbox.find('xmax').text, xmlbox.find('ymin').text,
             xmlbox.find('ymax').text)
        list2.append(BoundingBox(int(b[0]), int(b[2]), int(b[1]), int(b[3])))
        list1.append(str(cls_id))
        # file.write(",".join([str(a) for a in b]) + ',' + str(cls_id) + ' ')
    bbs = BoundingBoxesOnImage(list2, shape=img.shape)
    seq = iaa.Sequential([iaa.Resize((416, 416))], )
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)

    # image_after = bbs_aug.draw_on_image(image_aug, size=1, color=[0, 0, 255])
    # import matplotlib.pyplot as plt
    # plt.imshow(image_after)
    # plt.show()
    img = Image.fromarray(image_aug)
    img.save(output_path+basename_no_ext+'.jpg')
    for i in range(len(bbs_aug.bounding_boxes)):
        box = bbs_aug.bounding_boxes[i]
        obj = [box.x1,box.y1,box.x2,box.y2]
        file.write(",".join([str(int(a)) for a in obj]) + ',' + list1[i] + ' ')


cwd = getcwd()

for dir_path in dirs:
    output_path = 'D:/Files/GIT/CVFinalProject/data/VOC/'
    train_path = output_path+'train/'

    val_path = output_path+'valid/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    image_paths = getImagesInDir(dir_path)
    print(dir_path)
    list_train, list_val = train_test_split(image_paths, test_size=0.25, random_state=3030, shuffle=True)
    file_train = open(dir_path.replace('VOC2007/JPEGImages/', '') + 'train_annotation.txt', 'w')
    file_val = open(dir_path.replace('VOC2007/JPEGImages/', '') + 'valid_annotation.txt', 'w')

    for image_path in list_train:
        file_train.write(image_path[0].replace('../data/VOC2007/JPEGImages', 'data/VOC/train/').replace('\\', '') + ' ')
        convert_annotation(file_train, dir_path, train_path, image_path[0])
        file_train.write('\n')
    file_train.close()
    for image_path in list_val:
        file_val.write(image_path[0].replace('../data/VOC2007/JPEGImages', 'data/VOC/valid/').replace('\\', '') + ' ')
        convert_annotation(file_val, dir_path, val_path, image_path[0])
        file_val.write('\n')

    file_val.close()

    print("Finished processing")
