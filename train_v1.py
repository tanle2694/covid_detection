from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import time

from dataloaders.dataloader import CTImageLoader
from modeling.models.resnet import resnet50
from utils.configure_parse import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def validation(net, vali_loader, device):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(vali_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            print("----")
            print(predicted)
            print(targets)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    end = time.time()
    time_validation = end - start
    acc = correct / total * 100
    return loss, acc, time_validation

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", default="/home/tanlm/Downloads/covid_data/kfold/0/train.txt")
parser.add_argument("--validation_data", default="/home/tanlm/Downloads/covid_data/kfold/0/val.txt")
parser.add_argument("--root_folder", default="/home/tanlm/Downloads/covid_data/data")
parser.add_argument("--input_size", default=256)
parser.add_argument("--train_batch_size", default=32)
parser.add_argument("--vali_batch_size", default=32)
parser.add_argument("--seed", default=1)
parser.add_argument("--workers", default=3)
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--weight_decay", default=4e-5)
parser.add_argument("--max_iters", default=1000)
parser.add_argument("--epoch", default=1000)
parser.add_argument("--trainer_save_dir")
parser.add_argument("--exper_name")


args = parser.parse_args()


train_transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

validation_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = CTImageLoader(link_label_file=args.train_data, image_size=args.input_size, root_folder=args.root_folder,
                              transforms=train_transform)
validation_dataset = CTImageLoader(link_label_file=args.validation_data, image_size=args.input_size, root_folder=args.root_folder,
                                   transforms=validation_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.workers, drop_last=True)
vali_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.vali_batch_size, shuffle=False,
                                          num_workers=args.workers, drop_last=False)

criterion = torch.nn.CrossEntropyLoss()
net = resnet50(number_class=3, pretrained=True)
assert torch.cuda.is_available()
device = torch.device("cuda:0")
net = net.to(device)

optimzer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
iteration = 0
epoch = 0
# while iteration < args.max_iters:
for epoch in range(args.epoch):
    train_loss = 0
    correct = 0
    total = 0
    start = time.time()

    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimzer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimzer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end = time.time()
    time_epoch = end - start
    acc = correct / total * 100
    print("Training: Epoch: {:03d}  Iter:  {:06d}  Loss: {:0.4f}  Acc: {:0.2f} Time: {:2f}s/epoch".format(epoch,
                                                                                iteration, train_loss, acc, time_epoch))
    vali_loss, vali_acc, vali_time = validation(net, vali_loader, device)
    print("Validation: Epoch: {:03d}  Iter:  {:06d}  Loss: {:0.4f}  Acc: {:0.2f} Time: {:2f}s/epoch".format(epoch,
                                                                                                          iteration,
                                                                                                          vali_loss,
                                                                                                          vali_acc,
                                                                                                          vali_time))





