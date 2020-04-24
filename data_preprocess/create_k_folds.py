import os
import cv2
from random import shuffle

data_links = {"COVID_Positive_CT": [], "No_Finding_CT": [], "Nomal_Disease_CT": []}
label = {"COVID_Positive_CT": 0, "No_Finding_CT": 1, "Nomal_Disease_CT": 2}
root_folder = "/home/tanlm/Downloads/covid_data/data"


def write_to_file(list_link, output_file):
    with open(output_file, "w") as f:
        for key, links in list_link.items():
            for link in links:
                f.write("{}\t{}\n".format(link, label[key]))


for sub_folder in os.listdir(root_folder):
    for image_name in os.listdir(os.path.join(root_folder, sub_folder)):
        img_link = os.path.join(root_folder, sub_folder, image_name)
        # print(img_link)
        img = cv2.imread(img_link)
        assert img is not None
        data_links[sub_folder].append(os.path.join(sub_folder, image_name))


data_links_train = {}
data_links_val = {}
for key, list_link in data_links.items():
    shuffle(list_link)
    data_links[key] = list_link
save_folder = "/home/tanlm/Downloads/covid_data/kfold"
percent1= [0, 0.25, 0.5, 0.75]
percent2 = [0.25, 0.5, 0.75, 1]
for i in range(len(percent1)):
    for key, list_link in data_links.items():
        index1 = int(percent1[i] * len(list_link))
        index2 = int(percent2[i] * len(list_link))
        train = list_link[0: index1] + list_link[index2: ]
        vali = list_link[index1: index2]
        data_links_train[key] = train
        data_links_val[key] = vali
    if not os.path.exists(os.path.join(save_folder, str(i))):
        os.makedirs(os.path.join(save_folder, str(i)))
    write_to_file(data_links_train, os.path.join(save_folder, str(i), 'train.txt'))
    write_to_file(data_links_val, os.path.join(save_folder, str(i), 'val.txt'))