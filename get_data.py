import kaggle
import os
import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--kaggle_dataset',
                    type=str,
                    required=True)

ARGS = parser.parse_args()

DATA_FOLDER = 'data/'


def download_dataset(kaggle_dataset, data_folder):
    os.makedirs(data_folder, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(kaggle_dataset,
                                      path=data_folder,
                                      unzip=True,
                                      quiet=False)


def main():
    download_dataset(ARGS.kaggle_dataset, DATA_FOLDER)

if __name__ == '__main__':
    main()
