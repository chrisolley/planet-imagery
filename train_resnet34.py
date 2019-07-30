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
from models import Resnet34
from optimizer import lr_scheduler, create_optimizer
from logger import setup_logs
from helper_functions import save_model
from validation import validate

# directories
DATA_DIR = './data'
TEST_JPEG_DIR = os.path.join(DATA_DIR, 'test-jpg')
TEST_JPEG_ADD_DIR = os.path.join(DATA_DIR, 'test-jpg-additional')
LOG_DIR = './logs'
MODEL_DIR = './models'

# training parameters
BASE_OPTIMIZER = optim.Adam
INIT_LR_0 = 0.01
DIFF_LR_FACTORS = [9, 3, 1]
BATCH_SIZE = 64
EPOCHS = 40
MODEL = Resnet34(num_classes=17).cuda()

# training loop
def train(model, epochs, train_dl, val_dl):
    best_score = 0.0
    # create optimizer with differential learning rates
    optimizer = create_optimizer(model, BASE_OPTIMIZER, INIT_LR_0, DIFF_LR_FACTORS)
    iterations = epochs*len(train_dl)
    idx = 0
    for epoch in range(epochs):
        lr0 = lr_scheduler(epoch, 0.1, INIT_LR_0, 5)  # set base lr for this epoch
        optimizer = create_optimizer(model, BASE_OPTIMIZER, lr0, DIFF_LR_FACTORS)
        # training loop
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.cuda().float(), target.cuda().float()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            # unfreeze deeper layers sequentially
            if idx == int(0.1*iterations):
                model.unfreeze(1)
                logger.info("Iteration %d: Unfreezing group 1" % idx)
            if idx == int(0.2*iterations):
                model.unfreeze(0)
                logger.info("Iteration %d: Unfreezing group 0" % idx)
            if batch_idx % 100 == 0:
                logger.info("Epoch %d (Batch %d / %d)\t Train loss: %.3f" % \
                    (epoch+1, batch_idx, len(train_dl), loss.item()))
        # validation
        val_f2_score, val_loss = validate(model, val_dl, 0.2)
        logger.info("Epoch %d \t Validation loss: %.3f, F2 score: %.3f" % \
            (epoch+1, val_loss, val_f2_score))
        # model saving
        if val_f2_score > best_score:
            best_score = val_f2_score
            best_model_path = os.path.join(MODEL_DIR, 'model_resnet34_%d.pth' % \
                (100*val_f2_score))
            logger.info("Saving model to %s" % best_model_path)
            save_model(model, best_model_path)
    # return best_model_path


def main(model, run_name, partition, batch_size, epochs):
    # datasets
    train_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'), 
                            partition['inner_train'],
                            os.path.join(DATA_DIR, 'train_v2.csv'),
                            True)

    val_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'),
                        partition['validation'],
                        os.path.join(DATA_DIR, 'train_v2.csv'))

    # data loaders
    train_dl = DataLoader(train_ds,
                        batch_size=batch_size,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=True)

    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        num_workers=4,
                        pin_memory=True)

    train(model, epochs, train_dl, val_dl)


if __name__ == '__main__':
    # read in data splits
    with open(os.path.join(DATA_DIR, 'partition.p'), 'rb') as f:
        partition = pickle.load(f)
    # set up logs
    run_name = time.strftime("%Y-%m-%d_%H%M-") + "resnet34"
    logger = setup_logs(LOG_DIR, run_name)
    # train model
    main(MODEL, run_name, partition, BATCH_SIZE, EPOCHS)
