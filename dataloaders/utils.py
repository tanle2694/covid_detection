import cv2
import numpy as np
import re
import os


def get_imgs_labels(link_label_file, output_size, root_folder):
    with open(link_label_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    imgs = []
    labels = []
    images_link = []
    for line in lines:
        line_match = re.match(r"^(.*)\t([0|1|2])$", line)
        assert line_match is not None
        image_link = line_match.group(1)
        label = int(line_match.group(2))
        assert image_link is not None
        image_link = os.path.join(root_folder, image_link)
        images_link.append(image_link)
        assert os.path.exists(image_link)
        assert label is not None and (label in [0, 1, 2])
        image_preprocessed = read_padding_resize(image_link, output_size)
        imgs.append(image_preprocessed)
        labels.append(label)
    return imgs, labels, images_link



def padding_image(img_numpy):
    """
    TODO
    :param img_numpy:
    :return:
    """

    assert img_numpy.shape[2] == 3
    height, width = img_numpy.shape[0: 2]
    max_size = max(img_numpy.shape[0], img_numpy.shape[1])
    mask = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    start_x = (max_size - width) // 2
    start_y = (max_size - height) // 2
    mask[start_y: start_y + height, start_x: start_x + width, :] = img_numpy
    return mask

def read_padding_resize(image_link, output_size):
    img = cv2.imread(image_link)
    assert img is not None
    img_padding = padding_image(img)
    img_resize = cv2.resize(img_padding, (output_size, output_size))
    return img_resize
