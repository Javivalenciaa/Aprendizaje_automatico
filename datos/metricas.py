import numpy as np


def accuracy(y, y_pred):
    return (y == y_pred).mean()


def matriz_confusion(y, y_pred, n_clases):
    M = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y, y_pred):
        M[real, pred] += 1
    return M


def precision_recall_f1(M):
    diag = np.diag(M).astype(float)
    precision = diag / np.maximum(M.sum(axis=0), 1)
    recall = diag / np.maximum(M.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1
