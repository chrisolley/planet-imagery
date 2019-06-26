import os
import pandas as pd
import pickle

# directories
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

with open(os.path.join(DATA_DIR, 'partition.p'), 'wb') as f:
    pickle.dump(partition, f)
   