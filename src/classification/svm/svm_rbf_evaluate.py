"""
Multi-class Support Vector Machine (SVM) classifier on book covers.
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    # Load labels
    train = pd.read_csv('../../../data/book-dataset/book30-listing-train-train.csv')
    val = pd.read_csv('../../../data/book-dataset/book30-listing-train-val.csv')
    test = pd.read_csv('../../../data/book-dataset/book30-listing-test.csv', encoding="latin")

    # Load images with numpy
    X_train = np.load('../../../data/book-dataset/img_standardized/X_train.npy')
    Y_train = np.load('../../../data/book-dataset/img_standardized/Y_train.npy', allow_pickle=True).ravel().astype(
        np.uint8)

    X_val = np.load('../../../data/book-dataset/img_standardized/X_val.npy')
    Y_val = np.load('../../../data/book-dataset/img_standardized/Y_val.npy', allow_pickle=True).ravel().astype(np.uint8)

    X_test = np.load('../../../data/book-dataset/img_standardized/X_test.npy')
    Y_test = np.load('../../../data/book-dataset/img_standardized/Y_test.npy', allow_pickle=True).ravel().astype(np.uint8)

    clf = SVC(kernel="rbf", C=1000, gamma=1e-5)
    clf.fit(X_train, Y_train)

    acc = clf.score(X_val, Y_val)
    print("Validation:", acc)

    acc = clf.score(X_test, Y_test)
    print("Test:", acc)