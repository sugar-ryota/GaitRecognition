# %%

import numpy as np
import csv
import pandas as pd
import pickle
import random

from base.base_class import ConstrainedSMBase, MSMInterface, SMBase
from base.base import subspace_bases, mean_square_singular_values

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
        bases = subspace_bases(X, self.test_n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, :self.n_subdims]
        gramians = np.dot(dic.transpose(0, 2, 1), bases)

        return gramians
# %%


sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)

# %%


def predict(gallery, probe):

    TE_num = 100  # 変数R(TE特徴の数)
    sample_num = 5  # ランダムサンプリングする数

    # galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
    gallery_list = []
    num = 0
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/{gallery}km/tr_fc3.csv", header=None)
    df_array = np.array(df)
    add = int(df_array.shape[1]/sub_num)
    for i in range(sub_num):
        df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/{gallery}km/tr_fc3.csv",
                         header=None, usecols=[x for x in range(num, num+add)])
        gallery_list.append(df)
        num += add
    gallery_array = np.array(gallery_list)
    # gallery_trans = gallery_array.transpose(0, 2, 1)

    # %%
    ga_TE_feature = []
    for i in range(sub_num):
        TE_feature = []
        for j in range(TE_num):
            rand = [random.randint(0, add) for k in range(sample_num)]
            rand.sort()
            cnnfeature = []
            for k in rand:
                x = gallery_array[i][:, k-1]
                cnnfeature.extend(x)
            TE_feature.append(cnnfeature)
        TE_feature = np.array(TE_feature)
        ga_TE_feature.append(TE_feature.T)
    ga_TE_feature = np.array(ga_TE_feature)
    ga_TE_feature_trans = ga_TE_feature.transpose(0, 2, 1)
    save_path = "./accuracy/RTW_msm_fc3.txt"

    with open(save_path, mode='a') as file:
        file.write("\n")
        file.write("\n")
        file.write(f"gallery:{gallery}km probe:{probe}km")
        file.write("\n")
    # %%
    n_subdims = 25
    sum_acc = 0
    for p_num in range(1, 4):
        probe_list = []
        num = 0
        df = pd.read_csv(
            f"./cnnfeature/silhouette/cnnmodel/{gallery}km/ts_fc3{probe}km_{p_num}.csv", header=None)
        df_array = np.array(df)
        add = int(df_array.shape[1]/sub_num)
        for i in range(sub_num):
            df = pd.read_csv(
                f"./cnnfeature/silhouette/cnnmodel/{gallery}km/ts_fc3{probe}km_{p_num}.csv", header=None, usecols=[x for x in range(num, num+add)])
            probe_list.append(df)
            num += add
        probe_array = np.array(probe_list)
        # probe_trans = probe_array.transpose(0, 2, 1)
        pr_TE_feature = []
        for i in range(sub_num):
            TE_feature = []
            for j in range(TE_num):
                rand = [random.randint(0, add) for k in range(sample_num)]
                rand.sort()
                cnnfeature = []
                for k in rand:
                    x = probe_array[i][:, k-1]
                    cnnfeature.extend(x)
                TE_feature.append(cnnfeature)
            TE_feature = np.array(TE_feature)
            pr_TE_feature.append(TE_feature.T)
        pr_TE_feature = np.array(pr_TE_feature)
        pr_TE_feature_trans = pr_TE_feature.transpose(0, 2, 1)
        model = MutualSubspaceMethod(n_subdims=n_subdims)
        model.fit(ga_TE_feature_trans, y)
        model.n_subdims = 35
        pred = model.predict(pr_TE_feature_trans)
        print(f"pred: {pred}\n true: {y}\n")
        accuracy = (pred == y).mean()
        sum_acc += accuracy
        acc = sum_acc/p_num
        print(f"accuracy:{acc}")
        with open(save_path, mode='a') as file:
            file.write(f"accuracy:{acc}")
            file.write("\n")
