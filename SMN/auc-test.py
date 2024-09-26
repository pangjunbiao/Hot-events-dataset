import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import math


# ---自己按照公式实现
def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def auc_calculate(label, preds):
    y_argsort = label.argsort()
    pred_argsort = preds.argsort()
    y_lat = np.zeros(shape=(len(label), len(label)))
    pred_lat = np.zeros(shape=(len(label), len(label)))
    for i in range(len(label)):
        y_lat[i][y_argsort[i]] = 1
        # x_len = max(y_argsort[i], 200 - y_argsort[i])
        mean = pred_argsort[i]
        sigma = 8
        x = np.arange(0, 200)
        y = normal_distribution(x, mean, sigma)
        pred_lat[i] = y
    forest_auc = roc_auc_score(y_lat, pred_lat, multi_class='ovo')
    return forest_auc


if __name__ == '__main__':

    y = np.load('target_test.npy')[:20]
    y_argsort = y.argsort()+1
    pred = np.load('test_lambda_total_np.npy')[:20]
    pred_argsort = pred.argsort()+1
    k = 5
    m = 3
    ap_sum = 0
    for i in range(1,k):
        # rank(real) i rank(pred) pred_sort
        index = np.where(y_argsort == i)
        pred_sort = pred_argsort[index]
        if abs(pred_sort-i) > m:
            ap_sum = ap_sum+0
        else:
            ap_sum = ap_sum + i/pred_sort
            # ap_sum = ap_sum + 1
    ap = ap_sum/k
    # 预测的相关度定义
    # 选k个，rank(real)-rank(pred)
    # 相关度定义为  k-abs(rank(real)-rank(pred))
    DCG = 0
    IDCG = 0
    # 排序从1开始
    for i in range(1, k):
        index = np.where(y_argsort == i)
        pred_sort = pred_argsort[index]
        rel = 10 - abs(i-pred_sort)
        log2i = np.log2(i+1)
        DCG = DCG+log2i
        IDCG = IDCG + rel/log2i
    NDCG = DCG/IDCG
    y_lat = np.zeros(shape=(len(y), len(y)))
    pred_lat = np.zeros(shape=(len(y), len(y)))
    for i in range(len(y)):
        y_lat[i][y_argsort[i]] = 1
        x_len = max(y_argsort[i], 200 - y_argsort[i])
        mean = pred_argsort[i]
        sigma = 8
        x = np.arange(0, 200)
        y = normal_distribution(x, mean, sigma)
        pred_lat[i] = y
    forest_auc = roc_auc_score(y_lat, pred_lat, multi_class='ovo')
    print(forest_auc)
