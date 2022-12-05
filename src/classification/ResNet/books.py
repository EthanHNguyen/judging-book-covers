"""
Custom PyTorch dataset for the book-covers dataset.
"""
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset


class BookDataset(Dataset):
    """Book Covers dataset."""

    def __init__(self, csv, image_dir, transform=None):
        self.data = pd.read_csv(csv, encoding="latin-1").to_numpy()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_dir + self.data[idx][1])
        img = img / 255.0
        label = self.data[idx][5]

        if self.transform:
            img = self.transform(img)

        return img, label
