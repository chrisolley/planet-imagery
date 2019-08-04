import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
from models import VGG19
from optimizer import lr_scheduler, create_optimizer
from logger import setup_logs
from helper_functions import save_model
from validation import validate
import mlflow
import argparse
import tempfile

# directories
DATA_DIR = './data'
LOG_DIR = './logs'
MODEL_DIR = './models'

# fold parameters
SEED = 42
N_FOLDS = 4

# training parameters
parser = argparse.ArgumentParser(description='VGG16Training')
parser.add_argument('--init-lr-0', type=int, default=0.01,
                    help='initial learning rate for group 0 (default: 0.01')           
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size for training (default: 32')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train (default: 40)')
parser.add_argument('--patience', type=int, default=2,
                    help='number of epochs to wait before reducing lr (default: 2)')
args = parser.parse_args()

MODEL = VGG19(num_classes=17).cuda()
BASE_OPTIMIZER = optim.Adam
DIFF_LR_FACTORS = [9, 3, 1]

# training loop
def train(model, epochs, train_dl, val_dl, fold):
    best_score = 0.0
    lr0 = args.init_lr_0
    iterations = epochs*len(train_dl)
    idx = 0
    # create optimizer with differential learning rates
    optimizer = create_optimizer(MODEL, BASE_OPTIMIZER, args.init_lr_0, DIFF_LR_FACTORS)
    # set up lr schedule based on val loss
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
    for epoch in range(epochs):
        total_loss = 0
        # training loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.cuda().float(), target.cuda().float()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            total_loss += loss.item()
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
        # train loss
        train_loss = total_loss / len(train_dl)
        logger.info("Epoch %d\t Train loss: %.3f" % (epoch+1, train_loss))
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        # validation scores
        val_f2_score, val_loss = validate(model, val_dl, 0.2)
        # lr monitoring val_loss
        lr_scheduler.step(val_loss)
        logger.info("Epoch %d \t Validation loss: %.3f, F2 score: %.3f" % \
            (epoch+1, val_loss, val_f2_score))
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        mlflow.log_metric('val_f2_score', val_f2_score, step=epoch)
        # model saving
        if val_f2_score > best_score:
            best_score = val_f2_score
            best_model_path = os.path.join(MODEL_DIR, 'fold_%s' % fold, 'model_VGG19_%d.pth' % \
                (100*val_f2_score))
            logger.info("Saving model to %s" % best_model_path)
            save_model(model, best_model_path)
    # return best_model_path


def main(model, run_name, partition, batch_size, epochs, fold):
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

    train(model, epochs, train_dl, val_dl, fold)


if __name__ == '__main__':
    # data splitting
    train_file_names = list(os.listdir(os.path.join(DATA_DIR, 'train-jpg')))
    train_labels_df = pd.read_csv(os.path.join(DATA_DIR, 'train_v2.csv'))
    train_IDs = [f.split('.')[0] for f in train_file_names]

    # K fold
    kf = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(train_IDs)):
        if not os.path.exists(os.path.join(MODEL_DIR, 'fold_%s' % (fold+1) )):
            os.makedirs(os.path.join(MODEL_DIR, 'fold_%s' % (fold+1) ))
        inner_train_IDs = [train_IDs[index] for index in train_index]
        val_IDs = [train_IDs[index] for index in test_index]
        partition = {'inner_train': inner_train_IDs, 'validation': val_IDs}
        # set up logs
        run_name = time.strftime("%Y-%m-%d_%H%M-") + "VGG19"
        logger = setup_logs(LOG_DIR, run_name)
        # train model
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('model', MODEL.name)
            mlflow.log_param('fold', (fold+1) )
            with open('/tmp/lr.txt', 'w') as f:
                f.write('Optimizer:\t %s\n' % BASE_OPTIMIZER)
                f.write('LR Group Factors:\t %s\n' % str(DIFF_LR_FACTORS))
            mlflow.log_artifact('/tmp/lr.txt')
            for key, value in vars(args).items():
                mlflow.log_param(key, value)
            main(MODEL, run_name, partition, args.batch_size, args.epochs, fold+1)
        
