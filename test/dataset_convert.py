import sys

import numpy as np

sys.path.append('..')

import torchvision.transforms as tt
import torch
import os

from datasets.berlin import Berlin_single, Berlin_multi
from configuration.huston2013_multi_config import args

data_root = '../data/berlin'

modality_path_1 = os.path.join(data_root,
                               os.path.join('hsi', 'hsi' + '_X_train.mat'))
modality_path_2 = os.path.join(data_root,
                               os.path.join('sar', 'sar' + '_X_train.mat'))
label_train_path = os.path.join(data_root,
                                os.path.join('hsi', 'hsi' + '_Y_train.mat'))
berlin_train = Berlin_multi(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                            label_path=label_train_path,
                            data_transform=None,args=args)

modality_path_1 = os.path.join(data_root,
                               os.path.join('hsi', 'hsi' + '_X_test.mat'))
modality_path_2 = os.path.join(data_root,
                               os.path.join('sar', 'sar' + '_X_test.mat'))
label_train_path = os.path.join(data_root,
                                os.path.join('hsi', 'hsi' + '_Y_test.mat'))
berlin_test = Berlin_multi(modality_path_1=modality_path_1, modality_path_2=modality_path_2,
                           label_path=label_train_path,
                           data_transform=None,args=args)

hsi_path = '/home/data/shicaiwei/remote_sensing/berlin_decouple/train/hsi/'
sar_path = '/home/data/shicaiwei/remote_sensing/berlin_decouple/train/sar/'
for i in range(len(berlin_train)):
    sample = berlin_train[i]
    hsi, sar, label=sample['m_1'],sample['m_2'],sample['label']
    # print(label)
    np.save(hsi_path + f"{i}.npy", hsi)
    np.save(sar_path + f"{i}.npy", sar)

    with open('/home/data/shicaiwei/remote_sensing/berlin_decouple/train/' + 'labels.txt', 'a+', newline='') as label_f:
        label_f.write(f"{i},{label}")
        label_f.write('\n')



hsi_path = '/home/data/shicaiwei/remote_sensing/berlin_decouple/test/hsi/'
sar_path = '/home/data/shicaiwei/remote_sensing/berlin_decouple/test/sar/'
for i in range(len(berlin_test)):
    sample = berlin_test[i]
    hsi, sar, label=sample['m_1'],sample['m_2'],sample['label']
    np.save(hsi_path + f"{i}.npy", hsi)
    np.save(sar_path + f"{i}.npy", sar)

    with open('/home/data/shicaiwei/remote_sensing/berlin_decouple/test/' + 'labels.txt', 'a+', newline='') as label_f:
        label_f.write(f"{i},{label}")
        label_f.write('\n')