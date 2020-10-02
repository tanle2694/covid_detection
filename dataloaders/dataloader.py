import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import cv2
import numpy as np
from dataloaders.utils import get_imgs_labels

COVID_Positive = 0
No_Finding = 1
Normal_Disease = 2

class CTImageLoader(data.Dataset):
    def __init__(self, link_label_file, image_size, root_folder, transforms=None):
        super(CTImageLoader, self).__init__()
        self.labels = []
        self.images, self.labels, self.image_links = get_imgs_labels(link_label_file, image_size, root_folder)
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img = Image.fromarray(np.uint8(img[:, :, ::-1]))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class CTImageLoaderTest(data.Dataset):
    def __init__(self, link_label_file, image_size, root_folder, transforms=None):
        super(CTImageLoaderTest, self).__init__()
        self.labels = []
        self.images, self.labels, self.image_links = get_imgs_labels(link_label_file, image_size, root_folder)
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        link = self.image_links[index]
        img = Image.fromarray(np.uint8(img[:, :, ::-1]))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, link



if __name__ == "__main__":
    trans = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ct = CTImageLoader("/home/tanlm/Downloads/covid_data/train.txt", 224, "/home/tanlm/Downloads/covid_data/data", transforms=trans)
    ct_iter = iter(ct)
    trans = transforms.ToPILImage()
    for i in range(len(ct)):
        img, label = ct_iter.__next__()
        img = img *0.5 + 0.5
        img = trans(img)
        img = np.array(img, dtype=np.uint8)
        print(label)
        print(img.shape)
        print(type(img))
        cv2.imshow("df", np.uint8(img))
        cv2.waitKey(0)