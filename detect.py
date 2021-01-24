import matplotlib.pyplot as plt
import torch
from utils.tool import get_classes, detect_img
from model.centernet import Centernet
from PIL import Image

if __name__ == "__main__":
    input_shape = (416, 416, 3)
    test_annotation_path = 'data/test_annotation.txt'
    classes_path = 'data/class_name.txt'
    img_path = 'data/example.jpg'
    class_names = get_classes(classes_path)
    print('class_names = {}'.format(class_names))
    num_classes = len(class_names)

    Cuda = True

    # model_path = './logs/resnet18.pth'
    model_path = './logs/resnet50.pth'
    model = Centernet(num_classes, 'resnet50', pretrain=False)
    torch.load(model_path)
    model.load_state_dict(torch.load(model_path))

    if Cuda:
        model = model.cuda()
    img = Image.open(img_path)

    img = detect_img(model, img, class_names, Cuda)
    plt.imshow(img)
    plt.show()