import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataloader import Dataset, collate
from model.centernet import Centernet
from utils.tool import val_one_epoch, train_one_epoch


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = eval(f.readlines()[0])
    return class_names


if __name__ == "__main__":
    input_shape = (416, 416, 3)
    train_annotation_path = 'train_annotation.txt'
    valid_annotation_path = 'valid_annotation.txt'
    classes_path = 'class_name.txt'
    class_names = get_classes(classes_path)
    print('class_names = {}'.format(class_names))
    num_classes = len(class_names)
    pretrain = True
    Cuda = True

    model = Centernet(num_classes, pretrain)

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
    Freeze_Epoch = 30
    Epoch_Num = 100

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    train_dataset = Dataset(train_lines, input_shape, num_classes)
    val_dataset = Dataset(valid_lines, input_shape, num_classes)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                            drop_last=True, collate_fn=collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    model.freeze()

    for epoch in range(Init_Epoch, Epoch_Num):
        if epoch is Freeze_Epoch:
            lr = 1e-3
            optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
            model.unfreeze()

        train_loss = train_one_epoch(model, epoch, epoch_size, train_loader, Epoch_Num, Cuda, optimizer)
        print('Start Validation')
        val_loss = val_one_epoch(model, epoch, epoch_size_val, val_loader, Epoch_Num, Cuda)
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch_Num))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (train_loss, val_loss))

        print('Saving state, iter:', str(epoch + 1))
        if not Path('logs').exists():
            os.mkdir('logs')
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
            (epoch + 1), train_loss, val_loss))
        lr_scheduler.step(val_loss)
