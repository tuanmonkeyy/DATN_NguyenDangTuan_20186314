import glob2
import os

dataset_dir = "/media/tuan/study/DATN/dataset/DATASET/bird/birds"

files = []
tfiles = []
for ext in ["*.png", "*.jpeg", "*.jpg"]:
  image_files = glob2.glob(os.path.join("data/", ext))
  files += image_files
# you should have images with labels.txt in same folder
print("your images :",len(files))

from sklearn.model_selection import train_test_split

train_img_list, val_img_list = train_test_split(files, test_size=0.01, random_state=42)

print(len(train_img_list), len(val_img_list))

# with open('/media/tuan/study/DATN/dataset/train.txt', 'w') as f:
#   f.write('\n'.join(train_img_list) + '\n')

with open('/media/tuan/study/DATN/dataset/calib.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')

print("Done")
