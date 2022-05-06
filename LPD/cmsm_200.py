#!/usr/bin/env python
# coding: utf-8


# %%


import numpy as np
import csv
import pandas as pd
import pickle

from base.base_class import ConstrainedSMBase, MSMInterface
from base.base import subspace_bases, mean_square_singular_values


# %%


class ConstrainedMSM(MSMInterface, ConstrainedSMBase):
    """
    Constrained Mutual Subspace Method
    """

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)
        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """

        # bases, (n_dims, n_subdims)
        bases = subspace_bases(X, self.n_subdims)
        # bases, (n_gds_dims, n_subdims)
        bases = self._gds_projection(bases)

        # gramians, (n_classes, n_subdims, n_subdims)
        gramians = np.dot(self.dic.transpose(0, 2, 1), bases)

        return gramians


# %%
sub_num = 200
y = []
for i in range(sub_num):
    y.append(i)

# %%
# セルごとに実行する場合
# df_gallery = pd.read_csv(
#     "./OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-75_Gallery.csv", header=None)
# df_probe = pd.read_csv(
#     "./OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-75_Probe.csv", header=None)

df_gallery = pd.read_csv(
    "./LPD/OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-75_Gallery.csv", header=None)
df_probe = pd.read_csv(
    "./LPD/OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-75_Probe.csv", header=None)

df_ga_start = df_gallery.iloc[0:sub_num, 2]
df_ga_end = df_gallery.iloc[0:sub_num, 3]
df_pr_start = df_probe.iloc[0:sub_num, 2]
df_pr_end = df_probe.iloc[0:sub_num, 3]

df_ga_start = np.array(df_ga_start)
df_ga_end = np.array(df_ga_end)
df_pr_start = np.array(df_pr_start)
df_pr_end = np.array(df_pr_end)

df_ga_start = df_ga_start.flatten()
df_ga_end = df_ga_end.flatten()
df_pr_start = df_pr_start.flatten()
df_pr_end = df_pr_end.flatten()

ga_difference = []
pr_difference = []
# for i in range(3):
#     array = []
#     pr_difference.append(array)
for i in range(sub_num):
  ga_difference.append((df_ga_end[i]-df_ga_start[i])+1)
  pr_difference.append((df_pr_end[i]-df_pr_start[i])+1)

pr_difference_1 = []
pr_difference_2 = []
pr_difference_3 = []
for i in range(sub_num):
    a = int(pr_difference[i] / 3)
    if pr_difference[i] % 3 == 0:
        pr_difference_1.append(a)
        pr_difference_2.append(a)
        pr_difference_3.append(a)
    elif pr_difference[i] % 3 == 1:
        pr_difference_1.append(a+1)
        pr_difference_2.append(a)
        pr_difference_3.append(a)
    else:
        pr_difference_1.append(a+1)
        pr_difference_2.append(a+1)
        pr_difference_3.append(a)

pr_difference = []
pr_difference.append(pr_difference_1)
pr_difference.append(pr_difference_2)
pr_difference.append(pr_difference_3)

#   pr_difference[i] = int(pr_difference[i]/3)

# %%

#galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
gallery = []
num = 0
for i in range(sub_num):
    df = pd.read_csv("./LPD/cnnfeature/silhouette/cnnmodel/200.csv",
                     header=None, usecols=[x for x in range(num, num+ga_difference[i])])
    # gallery_list.append(df)
    df = np.array(df)
    df_t = df.T
    gallery.append(df_t)
    num += ga_difference[i]
# gallery_array = np.array(gallery_list)
# gallery_trans = gallery_array.transpose(0, 2, 1)


# %%


#probeの特徴に関して、それぞれの被験者ごとに配列に分ける
sum_acc = 0
for p_num in range(1, 4):  # probeデータは3つに分けているから一つずつpredictして平均をとる
    probe = []
    print(f"p_num = {p_num}")
    probe_list = []
    num = 0
    for i in range(sub_num):
        df = pd.read_csv(
            f"./LPD/cnnfeature/silhouette/cnnmodel/200_{p_num}.csv", header=None, usecols=[x for x in range(num, num+pr_difference[p_num-1][i])])
        # probe_list.append(df)
        df = np.array(df)
        df_t = df.T
        probe.append(df_t)
        num += pr_difference[p_num-1][i]
    # probe_array = np.array(probe_list)
    # probe_trans = probe_array.transpose(0, 2, 1)
    model = ConstrainedMSM(n_subdims=10, n_gds_dims=1000)
    model.fit(gallery, y)
    pred = model.predict(probe)
    print(f"pred: {pred}\n true: {y}\n")
    # accuracy = (pred == y).mean()
    accuracy = (pred == y).sum() / len(pred == y)
    print(f"accuracy:{accuracy}")
    sum_acc += accuracy
    acc = sum_acc/p_num
    print(f"total_accuracy:{acc}")





# %%
