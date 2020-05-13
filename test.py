from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import time
import cv2

from logger.logger import get_logger
from pathlib import Path
from dataloaders.dataloader import CTImageLoaderTest
from modeling.models.resnet import resnet50
from utils.configure_parse import ConfigParser
import modeling.loss as loss
from trainer.trainer import Trainer
import torchvision.transforms.functional as F
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def accuracy(output, target):
    with torch.no_grad():
        _, pred = torch.max(output, dim=1)
        # print(pred)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target)

def main(args):
    config = ConfigParser(args)
    cfg = config.config
    logger = get_logger(config.log_dir, "train")

    validation_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    validation_dataset = CTImageLoaderTest(link_label_file=cfg["validation_data"], image_size=cfg["input_size"],
                                       root_folder=cfg["root_folder"], transforms=validation_transform)

    vali_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg["vali_batch_size"], shuffle=False,
                                              num_workers=cfg["workers"], drop_last=False)

    model = resnet50(number_class=3, pretrained=True)
    checkpoint = torch.load(cfg['resume'])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    trans = transforms.ToPILImage()
    with torch.no_grad():
        for i, (data, target, links) in enumerate(vali_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, pred = torch.max(output, dim=1)
            for j in range(len(links)):
                print(links[j])
                print(pred[j].item())
                print(target[j].item())
                # if pred[j].item() == target[j].item():
                #     continue
                image = data[j] * 0.5 + 0.5
                image = trans(image.cpu())
                # image = image.cpu().data.numpy()
                # print(image)
                image = np.array(image)

                # cv2.imshow("as", image.astype(np.uint8))
                # cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data", default="/home/tanlm/Downloads/covid_data/kfold/0/train.txt")
    parser.add_argument("--validation_data", default="/home/tanlm/Downloads/covid_data/test/vali.txt")
    parser.add_argument("--root_folder", default="/home/tanlm/Downloads/covid_data/test/")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--vali_batch_size", default=32)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--weight_decay", default=4e-5)
    parser.add_argument("--max_iters", default=1000)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--resume", default="/home/tanlm/Downloads/covid_data/save_dir/models/exp0/0504_081615/model_best.pth")
    parser.add_argument("--trainer_save_dir", default="/home/tanlm/Downloads/covid_data/save_dir")
    parser.add_argument("--exper_name", default="exp0")
    args = parser.parse_args()
    main(args)