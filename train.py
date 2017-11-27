# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from config import *
from utils import *
from data import Fashion
from net import gen_model

data_transform_train = transforms.Compose([
        transforms.Scale(IMG_SIZE),
        transforms.RandomSizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform_test = transforms.Compose([
        transforms.Scale(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_loader = torch.utils.data.DataLoader(
    Fashion(train=True, transform=data_transform_train),
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    Fashion(train=False, transform=data_transform_test),
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

model = gen_model(freeze_param=True, model_path=DUMPED_MODEL).cuda(GPU_ID)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        if batch_idx and batch_idx % DUMP_INTERVAL == 0:
            print('Model saved to {}'.format(dump_model(model, epoch, batch_idx)))
    print('Model saved to {}'.format(dump_model(model, epoch)))


if __name__ == "__main__":
    for epoch in range(1, EPOCH + 1):
        train(epoch)

