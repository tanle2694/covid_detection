import cv2
import os
import pathlib
import numpy as np

root_dir = "/home/tanlm/Downloads/covid_data/test/"
train = pathlib.Path("/home/tanlm/Downloads/covid_data/test/train.txt").read_text()
train_dict = {}
vali = pathlib.Path("/home/tanlm/Downloads/covid_data/test/vali.txt").read_text()
vali_dict = {}

for line in train.split("\n")[:-1]:
    link, label = line.split("\t")
    train_dict[link] = label

for line in vali.split("\n")[:-1]:
    link, label = line.split("\t")
    vali_dict[link] = label

for key, value in train_dict.items():
    print(key)
    img = cv2.imread(os.path.join(root_dir, key))
    for key_val, value_val in vali_dict.items():
        img_val = cv2.imread(os.path.join(root_dir, key_val))
        print(key, key_val)
        if img_val.shape != img.shape:
            continue

        if np.sum((img - img_val)**2) ==0:
            print("exitzzzzz")
            exit()

