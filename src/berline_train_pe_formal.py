'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_CCR, HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross_PE_Formal, HSI_Lidar_Couple_Share
from src.berline_dataloader import berlin_multi_dataloader
from configuration.huston2013_multi_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_PE_Formal
from lib.processing_utils import get_file_list
import torch.optim as optim

import cv2
import numpy as np
import datetime
import random


def deeppix_main(args):
    args.class_num=8
    train_loader = berlin_multi_dataloader(train=True, args=args)
    test_loader = berlin_multi_dataloader(train=False, args=args)



    modality_to_channel = {'hsi': 244, 'sar': 4, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_Cross_PE_Formal(args, modality_1_channel, modality_2_channel)
    if torch.cuda.is_available():
        model.cuda()  #
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


    args.retrain = False
    train_base_multi_PE_Formal(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                               test_loader=test_loader,
                               args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
