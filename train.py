import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.centernet import Centernet
from utils.dataloader import Dataset, collate
from utils.tool import val_one_epoch, train_one_epoch, get_classes

if __name__ == "__main__":
    input_shape = (416, 416, 3)
    train_annotation_path = 'data/train_annotation.txt'
    valid_annotation_path = 'data/valid_annotation.txt'
    classes_path = 'data/class_name.txt'
    class_names = get_classes(classes_path)
    print('class_names = {}'.format(class_names))
    num_classes = len(class_names)
    pretrain = False
    Cuda = True
    backbone = 'resnet18'
    model = Centernet(num_classes, backbone, pretrain)

    if Cuda:
        model = model.cuda()

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(valid_annotation_path) as f:
        valid_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(valid_lines)

    lr = 1e-3
    Batch_size = 8
    Init_Epoch = 0
    Freeze_Epoch = 50
    Epoch_Num = 200

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

    train_dataset = Dataset(train_lines, input_shape, num_classes, augment=True)
    val_dataset = Dataset(valid_lines, input_shape, num_classes, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                            drop_last=True, collate_fn=collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    model.freeze()

    for epoch in range(Init_Epoch, Epoch_Num):
        if epoch is Freeze_Epoch:
            model.unfreeze()
            lr = 1e-4
            optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
            print("Unfreeze Model")

        train_loss = train_one_epoch(model, epoch, epoch_size, train_loader, Epoch_Num, Cuda, optimizer)
        print('Start Validation')
        val_loss = val_one_epoch(model, epoch, epoch_size_val, val_loader, Epoch_Num, Cuda)
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch_Num))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (train_loss, val_loss))

        print('Saving state, iter:', str(epoch + 1))
        if not Path('logs').exists():
            os.mkdir('logs')
        torch.save(model.state_dict(), 'logs/' + backbone + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
            (epoch + 1), train_loss, val_loss))
        lr_scheduler.step(val_loss)
