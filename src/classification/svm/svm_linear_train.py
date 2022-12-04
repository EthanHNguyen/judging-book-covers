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

    # Create search grid
    C_range = [1000]
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    best_score = 0
    best_params = {}
    best_clf = None
    for C in tqdm(C_range, desc="outer"):
        for gamma in tqdm(gamma_range, desc="inner", leave=False):
            clf = SVC(kernel="rbf", C=C, gamma=gamma)
            clf.fit(X_train, Y_train)
            acc = clf.score(X_val, Y_val)
            if acc > best_score:
                best_score = acc
                best_params = {"C": C, "gamma": gamma}
                best_clf = clf

    print("Best score: {:.2f}".format(best_score))
    print("Best parameters: {}".format(best_params))

    s = pickle.dumps(best_clf)
