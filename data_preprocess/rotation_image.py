import cv2
import os
import re
import shutil
root_folder = "/home/tanlm/data/covid/data/No_Finding_CT"
rot_folder = "/home/tanlm/data/covid/data/No_Finding_CT_rot"

for file_name in os.listdir(root_folder):
    print(file_name)
    rotation_match = re.match(r"^Rot(90|270|180).*$", file_name)
    if rotation_match is None:
        shutil.copy(os.path.join(root_folder, file_name), os.path.join(rot_folder, file_name))
        continue
    rotaion_degree = rotation_match.group(1)

    img = cv2.imread(os.path.join(root_folder, file_name))
    h, w = img.shape[:2]
    center = (w /2, h/2)
    assert img.shape[0] == img.shape[1]
    M = cv2.getRotationMatrix2D(center, -1 * int(rotaion_degree), 1.0)
    img_rotation = cv2.warpAffine(img, M, (h, w))
    cv2.imwrite(os.path.join(rot_folder, file_name), img_rotation)
    # cv2.imshow("origin", img)
    # cv2.imshow("rataion", img_rotation)
    # cv2.waitKey(0)
