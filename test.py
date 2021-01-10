import torch
from utils.tool import get_classes, detect_image
from model.centernet import Centernet
from torch.utils.data import DataLoader
from utils.dataloader import Dataset

if __name__ == "__main__":
    input_shape = (416, 416, 3)
    test_annotation_path = 'test_annotation.txt'
    classes_path = 'class_name.txt'
    class_names = get_classes(classes_path)
    print('class_names = {}'.format(class_names))
    num_classes = len(class_names)

    Cuda = True

    model_path = '.\logs\Epoch97-Total_Loss5.4814-Val_Loss5.9695.pth'
    model = Centernet(num_classes, 'resnet18', pretrain=False)
    model.load_state_dict(torch.load(model_path))

    if Cuda:
        model = model.cuda()


    with open(test_annotation_path) as f:
        test_lines = f.readlines()
    num_test = len(test_lines)

    test_dataset = Dataset(test_lines, input_shape, num_classes, augment=False)
    test_loader = DataLoader(test_dataset)
    detect_image(model, test_loader, Cuda)