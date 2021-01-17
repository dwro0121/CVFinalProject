import glob
import os
import xml.etree.ElementTree as ET
from os import getcwd

import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from sklearn.model_selection import train_test_split

dirs = ['data/VOC2007/JPEGImages/']
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class1 = ["cat", "dog", "person", "car", "horse"]

cwd = getcwd()

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append([filename])
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append([filename])

    return image_list

def get_annotation(dir_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml', encoding='UTF8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    list_bbox = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        if cls in class1:
            list_bbox.append([xmlbox.find('xmin').text, xmlbox.find('xmax').text, xmlbox.find('ymin').text,
             xmlbox.find('ymax').text, class1.index(cls)])
    return list_bbox

def process(file,output_path, img_path, bboxes):
    basename = os.path.basename(img_path)
    file.write(output_path + basename+ ' ')
    img = Image.open(img_path)
    img = np.array(img)
    bbs_list = []
    for bbox in bboxes:
        bbs_list.append(BoundingBox(int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])))
    bbs = BoundingBoxesOnImage(bbs_list, shape=img.shape)
    seq = iaa.Sequential([iaa.Resize((416, 416))], )
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    img = Image.fromarray(image_aug)
    img.save(output_path + basename)
    for i in range(len(bbs_aug.bounding_boxes)):
        box = bbs_aug.bounding_boxes[i]
        obj = [box.x1, box.y1, box.x2, box.y2]
        file.write(",".join([str(int(a)) for a in obj]) + ',' + str(bboxes[i][4]) + ' ')
    file.write("\n")

def process_testset(file,output_path, img_path, bboxes):
    basename = os.path.basename(img_path)
    basename_no_ext = os.path.splitext(basename)[0]

    gt_path = 'input/ground-truth/'+basename_no_ext+'.txt'
    file_gt = open(gt_path, 'w')

    file.write(output_path + basename+ ' ')
    img = Image.open(img_path)
    img = np.array(img)
    bbs_list = []
    for bbox in bboxes:
        bbs_list.append(BoundingBox(int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])))
    bbs = BoundingBoxesOnImage(bbs_list, shape=img.shape)
    seq = iaa.Sequential([iaa.Resize((416, 416))], )
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    img = Image.fromarray(image_aug)
    img.save(output_path + basename)
    for i in range(len(bbs_aug.bounding_boxes)):
        box = bbs_aug.bounding_boxes[i]
        obj = [box.x1, box.y1, box.x2, box.y2]
        file.write(",".join([str(int(a)) for a in obj]) + ',' + str(bboxes[i][4]) + ' ')
        file_gt.write(str(class1[bboxes[i][4]]))
        file_gt.write(' {} {} {} {}\n'.format(str(int(obj[0])), str(int(obj[1])), str(int(obj[2])), str(int(obj[3]))))
    file.write("\n")

def process_trainval():
    for dir_path in dirs:
        output_path = 'data/VOC/'
        train_path = output_path + 'train/'

        val_path = output_path + 'valid/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)

        image_paths = getImagesInDir(dir_path)
        file_train = open(dir_path.replace('VOC2007/JPEGImages/', '') + 'train_annotation.txt', 'w')
        file_val = open(dir_path.replace('VOC2007/JPEGImages/', '') + 'valid_annotation.txt', 'w')
        list_five = []
        for img_path in image_paths:
            list_bbox = get_annotation(dir_path,img_path[0])
            if len(list_bbox) > 0:
                list_five.append([img_path[0], list_bbox])
        list_train, list_val = train_test_split(list_five, test_size=0.20, random_state=3030, shuffle=True)
        for img_path, bboxes in list_train:
            process(file_train, train_path,img_path,bboxes)
        for img_path, bboxes in list_val:
            process(file_val, val_path,img_path,bboxes)
        class_names = open('data/class_name.txt', 'w')
        class_names.write("[\'"+"\',\'".join([str(a) for a in class1])+"\']")


def process_test():
    dirs = ['data/VOCdevkit/VOC2007/JPEGImages/']
    output_path = 'D:/Files/GIT/CVFinalProject/data/VOC/'
    test_path = output_path+'test/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for dir_path in dirs:
        image_paths = getImagesInDir(dir_path)
        file_test = open(dir_path.replace('VOCdevkit/VOC2007/JPEGImages/', '') + 'test_annotation.txt', 'w')
        list_five = []
        for image_path in image_paths:
            list_bbox = get_annotation(dir_path, image_path[0])
            if len(list_bbox) > 0:
                list_five.append([image_path[0], list_bbox])
        for img_path, bboxes in list_five:
            process_testset(file_test, test_path,img_path,bboxes)

process_test()