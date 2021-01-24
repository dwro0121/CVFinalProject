import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.dataloader import Dataset, collate


def cal_stdmean():
    train_annotation_path = 'train_annotation.txt'
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    print(len(train_lines))
    train_dataset = Dataset(train_lines, (416, 416, 3), len(train_lines), augment=False)
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0, collate_fn=collate, shuffle=False)
    nimages = 0
    mean = 0.
    std = 0.
    count = 0
    for data in train_loader:
        print(count)
        data = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in data]

        batch, batch_hms, batch_whs, batch_regs, batch_reg_masks = data
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        count += batch.size(0)
    mean /= nimages
    std /= nimages

    print(mean)
    print(std)


cal_stdmean()
