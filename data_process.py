import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

dataBasePath = 'data/Vehicles-OpenImages.v1-416x416.voc/'
saveBasePath = 'data/'
set_list = ['train', 'test', 'valid']


def process_one():
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'valid.txt'), 'w')
    handle_list = [ftrain, ftest, fval]
    for set_name, fhandle in zip(set_list, handle_list):
        xmlfilepath = dataBasePath + set_name
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        print('{}_num = {}'.format(set_name, len(total_xml)))
        for i in range(len(total_xml)):
            name = total_xml[i][:-4] + '\n'
            fhandle.write(name)
            # print('name = {}'.format(name))
            if set_name == 'train' or set_name == 'valid':
                ftrainval.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


# process_one()

def get_classes():
    classes = []
    for image_set in set_list:
        image_ids = open('{}{}.txt'.format(saveBasePath, image_set)).read().strip().split()

        for image_id in image_ids:
            in_file = open('{}{}/{}.xml'.format(dataBasePath, image_set, image_id), encoding='utf-8')
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.iter('object'):

                cls = obj.find('name').text
                if cls not in classes:
                    classes.append(cls)
    print('classes = {}'.format(classes))
    with open('class_name.txt', 'w') as classes_file:
        classes_file.write(str(classes))
    return classes


def convert_annotation(image_id, list_file, image_set):
    in_file = open('{}{}/{}.xml'.format(dataBasePath, image_set, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def process_two():
    classes = get_classes()

    for image_set in set_list:
        image_ids = open('{}{}.txt'.format(saveBasePath, image_set)).read().strip().split()
        list_file = open('{}_annotation.txt'.format(image_set), 'w')
        for image_id in image_ids:
            list_file.write('{}{}/{}.jpg'.format(dataBasePath, image_set, image_id))
            convert_annotation(image_id, list_file, image_set)
            list_file.write('\n')
        list_file.close()


def get_mean_and_std():
    image_set = 'train'
    image_ids = open('{}{}.txt'.format(saveBasePath, image_set)).read().strip().split()
    all_image = []
    for image_id in image_ids:
        image = Image.open('{}{}/{}.jpg'.format(dataBasePath, image_set, image_id))
        all_image.append((np.array(image, dtype=np.float32) / 255.))
    all_image = np.array(all_image)
    # print('all_image_shape = {}'.format(np.shape(all_image)))
    channel = np.shape(all_image)[-1]
    all_image = np.reshape(all_image, [-1, channel])

    print('mean = {}'.format(np.mean(all_image, axis=0)))
    print('std = {}'.format(np.std(all_image, axis=0)))
