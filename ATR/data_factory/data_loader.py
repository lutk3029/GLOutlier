import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])




def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SWaT0.5'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.6'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.7'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.8'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.9'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.4'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.3'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.2'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT0.1'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.1'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.2'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.3'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.4'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.5'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.6'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.7'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.8'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI0.9'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
