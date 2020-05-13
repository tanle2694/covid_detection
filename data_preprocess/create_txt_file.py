import os
import cv2
from random import shuffle

data_links = {"COVID_Positive_CT": [], "No_Finding_CT": [], "Nomal_Disease_CT": []}
label = {"COVID_Positive_CT": 0, "No_Finding_CT": 1, "Nomal_Disease_CT": 2}
root_folder = "/home/tanlm/Downloads/covid_data/test/train"


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
        data_links[sub_folder].append(os.path.join("train", sub_folder, image_name))



data_links_train = {}
data_links_val = {}

# for key, list_link in data_links.items():

write_to_file(data_links, '/home/tanlm/Downloads/covid_data/test/train/train.txt')
# write_to_file(data_links_val, '/home/tanlm/Downloads/covid_data/data/val.txt')

