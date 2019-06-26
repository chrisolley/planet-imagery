import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from helper_functions import rotate_cv, random_crop, center_crop, normalize


class PlanetDataset(Dataset):
    def __init__(self, img_folder, list_IDs, csv_path, transforms=False):
        self.list_IDs = list_IDs
        self.df = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.mlb = MultiLabelBinarizer()
        self.y = self.mlb.fit_transform([tag.split() for tag in self.df.tags])
        self.transforms = transforms

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        name = self.list_IDs[idx]
        img_path = os.path.join(self.img_folder, name + '.jpg')
        x = cv2.imread(img_path).astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
        if self.transforms:
            rdeg = (np.random.random() - 0.50) * 20
            x = rotate_cv(x, rdeg)
            x = random_crop(x)
            if np.random.random() > 0.5:
                x = np.fliplr(x).copy()
        else:
            x = center_crop(x)
        x = normalize(x)
        return np.rollaxis(x, 2), self.y[idx]
