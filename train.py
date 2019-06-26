import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import fbeta_score
import os
import pickle
from tqdm import tqdm
from time import ctime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from datasets import PlanetDataset
from models import Resnet34

DATA_DIR = os.path.abspath('data/')
TRAIN_JPEG_DIR = os.path.join(DATA_DIR, 'train-jpg')
TEST_JPEG_DIR = os.path.join(DATA_DIR, 'test-jpg')
TEST_JPEG_ADD_DIR = os.path.join(DATA_DIR, 'test-jpg-additional')

# data splitting
train_file_names = list(os.listdir(TRAIN_JPEG_DIR))
test_file_names = list(os.listdir(TEST_JPEG_DIR))
test_add_file_names = list(os.listdir(TEST_JPEG_ADD_DIR))
train_labels_df = pd.read_csv(os.path.join(DATA_DIR, 'train_v2.csv'))

train_IDs = [f.split('.')[0] for f in train_file_names]
test_IDs = [f.split('.')[0] for f in test_file_names]
test_add_IDs = [f.split('.')[0] for f in test_add_file_names]
inner_train_IDs, val_IDs = train_test_split(train_IDs,
                                            test_size=0.2,
                                            random_state=42)

partition = {'train': train_IDs, 'inner_train': inner_train_IDs,
             'validation': val_IDs, 'test': test_IDs, 'test_add': test_add_IDs}

# model
model = Resnet34(num_classes=17).cuda()

# datasets
train_ds = PlanetDataset(TRAIN_JPEG_DIR, partition['inner_train'], labels,
                         True)
val_ds = PlanetDataset(TRAIN_JPEG_DIR, partition['validation'], labels)

# data loaders
batch_size = 64
train_dl = DataLoader(train_ds,
                      batch_size=batch_size,
                      num_workers=4,
                      pin_memory=True,
                      shuffle=True)

val_dl = DataLoader(val_ds,
                    batch_size=batch_size,
                    num_workers=4,
                    pin_memory=True)
