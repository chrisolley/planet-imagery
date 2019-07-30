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
import time
from time import ctime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from datasets import PlanetDataset
from models import Net
from optimizer import lr_scheduler, create_optimizer
from logger import setup_logs
from helper_functions import save_model
from validation import validate


# directories
DATA_DIR = os.path.abspath('./data')
TEST_JPEG_DIR = os.path.join(DATA_DIR, 'test-jpg')
TEST_JPEG_ADD_DIR = os.path.join(DATA_DIR, 'test-jpg-additional')
LOG_DIR = './logs'
MODEL_DIR = './models'

# read in data splits
with open(os.path.join(DATA_DIR, 'partition.p'), 'rb') as f:
    partition = pickle.load(f)

# set up logs
run_name = time.strftime("%Y-%m-%d_%H%M-") + "custom_net"
logger = setup_logs(LOG_DIR, run_name)

# create model save dir
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# model
model = Net(num_classes=17).cuda()

# datasets
train_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'), 
                         partition['inner_train'],
                         os.path.join(DATA_DIR, 'train_v2.csv'),
                         True)

val_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'),
                       partition['validation'],
                       os.path.join(DATA_DIR, 'train_v2.csv'))

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

# training loop
def train(model, base_optimizer, init_lr, epochs, train_dl, val_dl):
    best_score = 0.0
    optimizer = base_optimizer(model.parameters(), lr=init_lr)
    for epoch in range(epochs):
        lr = lr_scheduler(epoch, 0.1, init_lr, 5)
        optimizer = base_optimizer(model.parameters(), lr=lr)
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.cuda().float(), target.cuda().float()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info("Epoch %d (Batch %d / %d)\t Train loss: %.3f" % \
                    (epoch+1, batch_idx, len(train_dl), loss.item()))
        val_f2_score, val_loss = validate(model, val_dl, 0.2)
        logger.info("Epoch %d \t Validation loss: %.3f, F2 score: %.3f" % \
            (epoch+1, val_loss, val_f2_score))
        if val_f2_score > best_score:
            best_score = val_f2_score
            file_path = os.path.join('models', 'model_net_%d.pth' % \
                (100*val_f2_score))
            logger.info("Saving model to %s" % file_path)
            save_model(model, file_path)

if __name__ == '__main__':
    # model
    model = Net(num_classes=17).cuda()
    # lr
    base_optimizer = optim.Adam
    init_lr = 0.001
    # training
    epochs = 40
    train(model, base_optimizer, init_lr, epochs, train_dl, val_dl)
