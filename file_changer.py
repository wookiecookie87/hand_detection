import cv2
import os
import sys


ROOT_DIR = os.path.abspath("../../")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/hands")
dataset_dir = os.path.join(DATASET_DIR, "train\mask")


files = os.listdir(dataset_dir)
save_dir = os.path.join(DATASET_DIR, "train\mask_binary")
os.makedirs(save_dir, exist_ok=True)

# img = cv2.imread(os.path.join(dataset_dir, "00000.png"))
#
# print(img.shape)




for file in files:
    print(file)
    img = cv2.imread(os.path.join(dataset_dir, file), 0)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(save_dir, file), thresh)


# VAL_IMAGE_IDS = [str(41220 + x) +".png" for x in range(38)]
#
# print(VAL_IMAGE_IDS)

# for filename in os.listdir(dataset_dir):
#     print(filename.split(".")[0])
#     src = os.path.join(dataset_dir, filename)
#     dst = os.path.join(dataset_dir, filename.split("_")[1])
#     os.rename(src, dst)
