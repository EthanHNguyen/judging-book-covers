"""
Multi-class Support Vector Machine (SVM) classifier on book covers.
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
import cv2
from tqdm import tqdm

# Load labels
train = pd.read_csv('../../data/book-dataset/book30-listing-train-train.csv')
val = pd.read_csv('../../data/book-dataset/book30-listing-train-val.csv')
test = pd.read_csv('../../data/book-dataset/book30-listing-test.csv', encoding="latin")

# Load images with numpy
X_train = np.load('../../data/book-dataset/img_standardized/X_train.npy')
Y_train = np.load('../../data/book-dataset/img_standardized/Y_train.npy', allow_pickle=True).ravel().astype(np.uint8)


# X_train = X_train[:1000]
# Y_train = Y_train[:1000]

X_val = np.load('../../data/book-dataset/img_standardized/X_val.npy')
Y_val = np.load('../../data/book-dataset/img_standardized/Y_val.npy', allow_pickle=True).ravel().astype(np.uint8)

# Fit images using SVM
clf = LinearSVC(verbose=1, max_iter=1000)
clf.fit(X_train, Y_train)

# Predict on validation set
acc = clf.score(X_val, Y_val)
print("Validation accuracy: " + str(acc))