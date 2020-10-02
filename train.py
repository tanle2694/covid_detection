from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import time

from logger.logger import get_logger
from pathlib import Path
from dataloaders.dataloader import CTImageLoader
from modeling.models.resnet import resnet50
from utils.configure_parse import ConfigParser
import modeling.loss as loss
from trainer.trainer import Trainer

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



def main(args):
    config = ConfigParser(args)
    cfg = config.config
    logger = get_logger(config.log_dir, "train")
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

    train_dataset = CTImageLoader(link_label_file=cfg["train_data"], image_size=cfg["input_size"],
                                  root_folder=cfg["root_folder"], transforms=train_transform)
    validation_dataset = CTImageLoader(link_label_file=cfg["validation_data"], image_size=cfg["input_size"],
                                       root_folder=cfg["root_folder"], transforms=validation_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["train_batch_size"], shuffle=True,
                                               num_workers=cfg["workers"], drop_last=True)
    vali_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg["vali_batch_size"], shuffle=False,
                                              num_workers=cfg["workers"], drop_last=False)

    model = resnet50(number_class=3, pretrained=True)

    criterion = getattr(loss, 'cross_entropy')
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    metrics_name = ["accuracy"]
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, nb_epochs=config['epoch'],
                      valid_loader=vali_loader, logger=logger, log_dir=config.save_dir, metrics_name=metrics_name,
                      resume=config['resume'], save_dir=config.save_dir)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="/home/tanlm/Downloads/covid_data/kfold/0/train.txt")
    parser.add_argument("--validation_data", default="/home/tanlm/Downloads/covid_data/kfold/0/val.txt")
    parser.add_argument("--root_folder", default="/home/tanlm/Downloads/covid_data/data")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--train_batch_size", default=64)
    parser.add_argument("--vali_batch_size", default=64)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--weight_decay", default=4e-5)
    parser.add_argument("--max_iters", default=1000)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--resume", default="")
    parser.add_argument("--trainer_save_dir", default="/home/tanlm/Downloads/covid_data/save_dir")
    parser.add_argument("--exper_name", default="fold_0")
    args = parser.parse_args()
    main(args)