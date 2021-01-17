import torch
from utils.tool import get_classes, detect_image
from model.centernet import Centernet
from torch.utils.data import DataLoader
from utils.dataloader import Dataset

if __name__ == "__main__":
    input_shape = (416, 416, 3)
    test_annotation_path = 'data/test_annotation.txt'
    classes_path = 'data/class_name.txt'
    class_names = get_classes(classes_path)
    print('class_names = {}'.format(class_names))
    num_classes = len(class_names)

    Cuda = True

    # model_path = './logs/resnet18Epoch100-Total_Loss3.7628-Val_Loss5.4680.pth'
    model_path = './logs/resnet50Epoch100-Total_Loss3.7447-Val_Loss5.4505.pth'
    model = Centernet(num_classes, 'resnet50', pretrain=False)
    torch.load(model_path)
    model.load_state_dict(torch.load(model_path))

    if Cuda:
        model = model.cuda()


    with open(test_annotation_path) as f:
        test_lines = f.readlines()
    num_test = len(test_lines)

    test_dataset = Dataset(test_lines, input_shape, num_classes, augment=True, tvt='test')
    test_loader = DataLoader(test_dataset)
    detect_image(model, test_loader,class_names, Cuda)