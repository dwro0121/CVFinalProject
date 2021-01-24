import torch
from torch.utils.data import DataLoader

from model.centernet import Centernet
from utils.dataloader import Dataset
from utils.tool import get_classes, test_model

if __name__ == "__main__":
    input_shape = (416, 416, 3)
    test_annotation_path = 'data/test_annotation.txt'
    classes_path = 'data/class_name.txt'
    class_names = get_classes(classes_path)
    print('class_names = {}'.format(class_names))
    num_classes = len(class_names)

    Cuda = True

    backbone = 'resnet50'

    model_path = './logs/centernet_'+backbone.replace('net','')+'.pth'
    model = Centernet(num_classes, 'backbone', pretrain=False)
    torch.load(model_path)
    model.load_state_dict(torch.load(model_path))

    if Cuda:
        model = model.cuda()

    with open(test_annotation_path) as f:
        test_lines = f.readlines()
    num_test = len(test_lines)

    test_dataset = Dataset(test_lines, input_shape, num_classes, augment=True, tvt='test')
    test_loader = DataLoader(test_dataset)
    test_model(model, test_loader, class_names, Cuda)
