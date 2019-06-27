from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import time
from train_resnet34 import train
from models import Resnet34
from sklearn.model_selection import KFold
from logger import setup_logs
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import PlanetDataset

DATA_DIR = os.path.abspath('data/')
SAVE_DIR = './snapshots'
train_file_names = list(os.listdir(os.path.join(DATA_DIR, 'train-jpg')))
test_file_names = list(os.listdir(os.path.join(DATA_DIR, 'test-jpg')))
test_add_file_names = list(os.listdir(os.path.join(DATA_DIR, 'test-jpg-additional')))

train_IDs = [f.split('.')[0] for f in train_file_names]
test_IDs = [f.split('.')[0] for f in test_file_names]
test_add_IDs = [f.split('.')[0] for f in test_add_file_names]

batch_size = 64

run_name = time.strftime("%Y-%m-%d_%H%M-") + "prediction"
logger = setup_logs(SAVE_DIR, run_name)


def make_test_predictions(model, test_dl):
    predictions = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dl, desc='Test prediction', total=len(test_dl)):
            data = data.cuda().float()
            pred = model(data)
            predictions.append(F.sigmoid(pred).cpu().numpy())
        return np.vstack(predictions)


def make_tta_prediction(model, test_dl, test_dl_aug, n_tta):
    predictions = make_test_predictions(model, test_dl)
    tta_predictions = []
    for _ in tqdm(range(n_tta), desc='Testing time augmentation'):
        tta_predictions.append(make_test_predictions(model, test_dl_aug))
    tta_predictions = tta_predictions + [predictions]
    tta_predictions = np.mean(tta_predictions, axis=0)
    return tta_predictions


test_ds = PlanetDataset(os.path.join(DATA_DIR, 'test-jpg'),
                        test_IDs,
                        os.path.join(DATA_DIR, 'sample_submission_v2.csv'))

test_add_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'),
                            test_add_IDs,
                            os.path.join(DATA_DIR, 'sample_submission_v2.csv'))

test_ds_aug = PlanetDataset(os.path.join(DATA_DIR, 'test-jpg'),
                            test_IDs,
                            os.path.join(DATA_DIR, 'sample_submission_v2.csv'),
                            True)

test_add_ds_aug = PlanetDataset(os.path.join(DATA_DIR, 'test-jpg'),
                                test_add_IDs,
                                os.path.join(DATA_DIR, 'sample_submission_v2.csv'),
                                True)

test_dl = DataLoader(test_ds,
                     batch_size=batch_size,
                     num_workers=4,
                     pin_memory=True)

test_dl_aug = DataLoader(test_ds_aug,
                         batch_size=batch_size,
                         num_workers=4,
                         pin_memory=True)

test_add_dl = DataLoader(test_add_ds,
                         batch_size=batch_size,
                         num_workers=4,
                         pin_memory=True)

test_add_dl_aug = DataLoader(test_add_ds_aug,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True)


def k_fold_predict(n_splits):
    overall_pred, overall_pred_add = [], []
    model = Resnet34(num_classes=17).cuda()
    kf = KFold(n_splits=4)
    for i, (train_index, val_index) in enumerate(kf.split(train_IDs)):
        logger.info("Fold %d" % (i+1))
        inner_train_IDs = list(np.array(train_IDs)[train_index])
        val_IDs = list(np.array(train_IDs)[val_index])
        partition = {'inner_train': inner_train_IDs, 'validation': val_IDs}

        train_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'), 
                         partition['inner_train'],
                         os.path.join(DATA_DIR, 'train_v2.csv'),
                         True)

        val_ds = PlanetDataset(os.path.join(DATA_DIR, 'train-jpg'),
                            partition['validation'],
                            os.path.join(DATA_DIR, 'train_v2.csv'))

        train_dl = DataLoader(train_ds,
                     batch_size=batch_size,
                     num_workers=4,
                     pin_memory=True)
        
        val_dl = DataLoader(val_ds,
                     batch_size=batch_size,
                     num_workers=4,
                     pin_memory=True)

        best_model_path = train(model, 0.01, 30, train_dl, val_dl)
        logger.info("Training complete")
        best_model = Resnet34(num_classes=17).cuda()
        load_model(best_model, best_model_path)
        logger.info("Loading best model")
        logger.info("Making TTA predictions")
        tta_pred = make_tta_prediction(best_model, test_dl, test_dl_aug, 4)
        tta_add_pred = make_tta_prediction(model, test_add_dl, test_add_dl_aug, 4)
        logger.info("TTA predictions complete")
        overall_pred.append(tta_pred)
        overall_pred_add.append(tta_add_pred)
    overall_pred = np.mean(overall_pred, axis=0)
    overall_pred_add = np.mean(overall_pred_add, axis=0)
    return overall_pred, overall_pred_add

if __name__ == '__main__':
    threshold = 0.2
    tta_predictions, tta_predictions_add = k_fold_predict(n_splits=4)
    logger.info('Thresholding predictions')
    tta_predictions_hard = tta_predictions > threshold
    tta_predictions_add_hard = tta_add_predictions > threshold
    tags = test_ds.mlb.inverse_transform(tta_predictions_hard)
    tags_add = test_ds.mlb.inverse_transform(tta_predictions_add_hard)
    logger.info('Creating prediction dataframe')
    test_result_df = pd.DataFrame({'image_name': partition['test'], 'tags': [' '.join(t) for t in tags]})
    test_add_result_df = pd.DataFrame({'image_name': partition['test_add'], 'tags': [' '.join(t) for t in tags_add]})
    test_result_df = pd.concat([test_result_df, test_add_result_df])
    logger.info('Saving predictions to csv')
    submission_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission_v2.csv'))
    submission_df.drop('tags', axis=1, inplace=True)
    submission_df = submission_df.merge(test_result_df, on='image_name')
    submission_df.to_csv('out_kfold4.csv', index=False)




