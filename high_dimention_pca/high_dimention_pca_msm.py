#!/usr/bin/env python
# coding: utf-8

# %%


import numpy as np
import pandas as pd

from base.base_class import MSMInterface, SMBase
from base.base import subspace_bases,specific_subspace_bases


# %%


"""
Mutual Subspace Method
"""


class MutualSubspaceMethod(MSMInterface, SMBase):
    """
    Mutual Subspace Method
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
        # bases = subspace_bases(X, self.test_n_subdims)
        bases = specific_subspace_bases(X)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, :self.n_subdims]
        gramians = np.dot(dic.transpose(0, 2, 1), bases)
        # bases_array = np.array(bases)
        # dic_array = np.array(dic)
        # gramians_array = np.array(gramians)
        # print(f'bases_shape = {bases_array.shape}')
        # print(f'dic_shape = {dic_array.shape}')
        # print(f'gramians_shape = {gramians_array.shape}')
        return gramians


# %%


sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)


# %%
train = 2
test = 3

#galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
num = 0
df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/{train}km/tr.csv", header=None)
df_array = np.array(df)
add = int(df_array.shape[1]/sub_num)
for i in range(sub_num):
    df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/{train}km/tr.csv",
                     header=None, usecols=[x for x in range(num, num+add)])
    gallery_list.append(df)
    num += add
gallery_array = np.array(gallery_list)
gallery_trans = gallery_array.transpose(0, 2, 1)


# %%


#probeの特徴に関して、それぞれの被験者ごとに配列に分ける
sum_acc = 0
for p_num in range(1, 4):  # probeデータは3つに分けているから一つずつpredictして平均をとる
    print(f"p_num = {p_num}")
    probe_list = []
    num = 0
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/{train}km/ts{test}km_{p_num}.csv", header=None)
    df_array = np.array(df)
    add = int(df_array.shape[1]/sub_num)
    for i in range(sub_num):
        df = pd.read_csv(
            f"./cnnfeature/silhouette/cnnmodel/{train}km/ts{test}km_{p_num}.csv", header=None, usecols=[x for x in range(num, num+add)])
        probe_list.append(df)
        num += add
    probe_array = np.array(probe_list)
    probe_trans = probe_array.transpose(0, 2, 1)
    model = MutualSubspaceMethod(n_subdims=20)
    model.fit(gallery_trans, y)
    model.n_subdims = 10
    pred = model.predict(probe_trans)
    print(f"pred: {pred}\n true: {y}\n")
    accuracy = (pred == y).mean()
    sum_acc += accuracy
    acc = sum_acc/p_num
    print(f"accuracy:{acc}")
