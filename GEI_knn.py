# %%

import csv

import os

import numpy as np

import pandas as pd

import tensorflow as tf

import glob

import openpyxl as excel

from skimage import data

from PIL import Image

from pathlib import Path

import pickle

import cv2

import matplotlib

import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import RMSprop

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

# %%
sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)

# %%
gallery_list = []
ga_df = pd.read_csv("./gei/cnnfeature/3km/tr.csv", header=None)
ga_df = np.array(ga_df)
ga_df_T = ga_df.T
# PCAによる次元削減
pca = PCA(n_components=2)
pca.fit(ga_df_T)
X_train = pca.transform(ga_df_T)

# %%
for p_num in range(1, 4):  # probeデータは3つに分けているから一つずつpredictして平均をとる
    print(f"p_num = {p_num}")
    probe_list = []
    pr_df = pd.read_csv(f"./gei/cnnfeature/3km/ts3km_{p_num}.csv", header=None)
    pr_df = np.array(pr_df)
    pr_df_T = pr_df.T

    # PCAによる次元削減
    pca = PCA(n_components=2)
    pca.fit(pr_df_T)
    X_test = pca.transform(pr_df_T)

    # インスタンス
    knn = KNeighborsClassifier(n_neighbors=1)

    # モデル学習
    knn.fit(ga_df_T, y)

    pred = knn.predict(pr_df_T)
    print(pred)

    print("正解率: " + str(round(accuracy_score(y, pred), 3)))
# # %%
# gallery_list = []
# num = 0
# for i in range(sub_num):
#     df = pd.read_csv("./gei/cnnfeature/2km/tr.csv",
#                      header=None, usecols=[x for x in range(num, num+1)])
#     gallery_list.append(df)
#     num += 1

# gallery_list = np.array(gallery_list)
# # print(gallery_list[0].shape)
# # %%
# for p_num in range(1, 4):  # probeデータは3つに分けているから一つずつpredictして平均をとる
#     print(f"p_num = {p_num}")
#     sum_acc = 0
#     probe_list = []
#     num = 0
#     for i in range(sub_num):
#         df = pd.read_csv(
#             f"./gei/cnnfeature/2km/ts2km_{p_num}.csv", header=None, usecols=[x for x in range(num, num+1)])
#         probe_list.append(df)
#         num += 1
#     probe_list = np.array(probe_list)

#     # インスタンス
#     knn = KNeighborsClassifier(n_neighbors=1)

#     # モデル学習
#     for i in range(sub_num):
#         knn.fit(gallery_list[i], y[i])

#         pred = knn.predict(probe_list[i])

#     print("正解率: " + str(round(accuracy_score(y, pred), 1)))


# %%
