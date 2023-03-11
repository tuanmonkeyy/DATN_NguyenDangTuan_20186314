import os
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

data_dir = '/media/tuan/study/DATN/dataset/tapcon_de_thu_pp_TL/data'
label_dir = '/media/tuan/study/DATN/dataset/tapcon_de_thu_pp_TL/data'
data_dir = Path(data_dir)
label_dir = Path(label_dir)

sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
# set fonttype
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

#%matplotlib inline


class_counter = {'train': Counter(), 'val': Counter()}
class_freqs = {}
files_train = []
files_val = []
with open(Path('/media/tuan/study/DATN/dataset/tapcon_de_thu_pp_TL') / 'train.txt', 'r') as f:
    for line in f:
        newline = line.strip()
        files_train.append(newline)
        #image_id = line.split('/')[-1].split('.+?/(?=[^/]+$)')[-1]
        #basename = os.path.basename(line)
        #image_id = os.path.splitext(basename)[0]
        #df = np.loadtxt(label_dir / f'{image_id}.txt',ndmin=2)
        #class_counter['train'].update(df[:, 0].astype(int))
for fil in files_train:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    df = np.loadtxt(label_dir / f'{filename}.txt',ndmin=2)
    class_counter['train'].update(df[:, 0].astype(int))
# get class freqs
total = sum(class_counter['train'].values()) #tong so lan xuat hien
print("train")
for v in class_counter['train'].items():
    print(v)
print("\n")
class_freqs['train'] = {k: v / total for k, v in class_counter['train'].items()}
        
with open(Path('/media/tuan/study/DATN/dataset/tapcon_de_thu_pp_TL') / 'val.txt', 'r') as f:
    for line in f:
        newline = line.strip()
        files_val.append(newline)
        # image_id = line.split('/')[-1].split('.+?/(?=[^/]+$)')[-1]
        # basename = os.path.basename(line)
        # image_id = os.path.splitext(basename)[0]
        # df = np.loadtxt(label_dir / f'{image_id}.txt',ndmin=2)
        # class_counter['val'].update(df[:, 0].astype(int))
for fil in files_val:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    df = np.loadtxt(label_dir / f'{filename}.txt',ndmin=2)
    class_counter['val'].update(df[:, 0].astype(int))
# get class freqs
total = sum(class_counter['val'].values())
print("val\n")
for v in class_counter['val'].items():
    print(v)
print("\n")
class_freqs['val'] = {k: v / total for k, v in class_counter['val'].items()}

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(range(5), [class_freqs['train'][i] for i in range(5)], color='navy', label='train');
ax.plot(range(5), [class_freqs['val'][i] for i in range(5)], color='tomato', label='val');
ax.legend();
ax.set_xlabel('Class ID');
ax.set_ylabel('Class Frequency');
#plt.imshow(ax)
plt.show()
