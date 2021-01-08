import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.centernet import Centernet
from utils.losses import focal_loss, l1_loss
from dataloader import CenternetDataset, centernet_dataset_collate


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = eval(f.readlines()[0])
    return class_names


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            optimizer.zero_grad()

            ret = net(batch_images)
            hm, wh, offset = ret['hm'], ret['sizes'], ret['offsets']
            # print('hm_shape = {},batch_hms_shape = {}'.format(hm.shape,batch_hms.shape))
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = l1_loss(offset, batch_regs, batch_reg_masks)

            loss = c_loss + wh_loss + off_loss

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += wh_loss.item() + off_loss.item()

            loss.backward()
            optimizer.step()

            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)

            start_time = time.time()

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                ret = net(batch_images)
                hm, wh, offset = ret['hm'], ret['sizes'], ret['offsets']

                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = l1_loss(offset, batch_regs, batch_reg_masks)
                loss = c_loss + wh_loss + off_loss
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    if not Path('logs').exists():
        os.mkdir('logs')
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    return val_loss / (epoch_size_val + 1)


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

    model = Centernet(num_classes,pretrain)

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(valid_annotation_path) as f:
        valid_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(valid_lines)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = CenternetDataset(train_lines, input_shape, num_classes)
        val_dataset = CenternetDataset(valid_lines, input_shape, num_classes)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                             drop_last=True, collate_fn=centernet_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        model.freeze()

        for epoch in range(Init_Epoch, Freeze_Epoch):
            val_loss = fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step(val_loss)

    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = CenternetDataset(train_lines, input_shape, num_classes)
        val_dataset = CenternetDataset(valid_lines, input_shape, num_classes)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                         drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                             drop_last=True, collate_fn=centernet_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        model.unfreeze()

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            val_loss = fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step(val_loss)