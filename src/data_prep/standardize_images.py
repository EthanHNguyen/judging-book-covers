"""
Standardize and downscale images for svm.
"""

import pandas as pd
import numpy as np
import cv2
import multiprocessing as mp

# Load labels
train = pd.read_csv('../../data/book-dataset/book30-listing-train-train.csv')
val = pd.read_csv('../../data/book-dataset/book30-listing-train-val.csv')
test = pd.read_csv('../../data/book-dataset/book30-listing-test.csv', encoding="latin")

train = train.to_numpy()
val = val.to_numpy()
test = test.to_numpy()


# Load images with multiprocessing
def load_image(row):
    img = cv2.imread('../../data/book-dataset/img/' + row[1])
    img = cv2.resize(img, (64, 64))
    return img


if __name__ == '__main__':
    train_mean, train_std = None, None
    for data, name in [(train, "train"), (val, "val"), (test, "test")]:
        # Load images
        with mp.Pool(mp.cpu_count() * 2) as pool:
            X = pool.map(load_image, data)
        X = np.array(X)
        Y = data[:, 5]

        # Calculate mean and std
        if train_mean is None and train_std is None:
            train_mean = np.mean(X, axis=(0, 1, 2))
            train_std = np.std(X, axis=(0, 1, 2))

        # Standardize images
        X = (X - train_mean) / train_std

        # Flatten image
        X = X.reshape(X.shape[0], -1)

        # Save images
        np.save('../../data/book-dataset/img_standardized/X_' + name + '.npy', X)
        np.save('../../data/book-dataset/img_standardized/Y_' + name + '.npy', Y)

        # Print size of X and Y
        print("X_" + name + ".npy: " + str(X.size) + " bytes")
        print("Y_" + name + ".npy: " + str(Y.size) + " bytes")

        # Print progress
        print("Finished standardizing " + name + " images.")
