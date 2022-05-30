# %%

import numpy as np
import pandas as pd
import random
import math

from base.base_class import ConstrainedSMBase, MSMInterface
from base.base import subspace_bases

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


sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)

# %%

TE_num = 101  # 変数R(TE特徴の数)
sample_num = 5  # ランダムサンプリングする数
m = 100  # サンプル間隔


# %%

# galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
num = 0
df = pd.read_csv("./cnnfeature/silhouette/cnnmodel/2km/tr_ex.csv", header=None)
df_array = np.array(df)
add = int(df_array.shape[1]/sub_num)
for i in range(sub_num):
    df = pd.read_csv("./cnnfeature/silhouette/cnnmodel/2km/tr_ex.csv",
                     header=None, usecols=[x for x in range(num, num+add)])
    gallery_list.append(df)
    num += add
gallery_array = np.array(gallery_list)  # gallery_array.shape = ()

# シフト量の計算
s = (gallery_array.shape[2] - m) / (TE_num - 1)  # シフト量s
# %%
ga_TE_feature = []
for i in range(sub_num):
    TE_feature = []
    start = 0
    last = m
    for j in range(TE_num):
        rand = [random.randint(int(start), int(last))
                for k in range(sample_num)]
        rand.sort()
        cnnfeature = []
        for k in rand:
            x = gallery_array[i][:, k-1]
            cnnfeature.extend(x)  # ランダムに取り出したベクトルを一列につなげる
        TE_feature.append(cnnfeature)
        # if(j % 4 == 0):
        #     ceil = math.ceil(s)
        #     start += ceil
        #     last += ceil
        # else:
        #     floor = math.floor(s)
        #     start += floor
        #     last += floor
        start += s
        last += s
        # if(last > gallery_array.shape[2]):
        #     last = gallery_array.shape[2]
    TE_feature = np.array(TE_feature)
    ga_TE_feature.append(TE_feature.T)
ga_TE_feature = np.array(ga_TE_feature)
ga_TE_feature_trans = ga_TE_feature.transpose(0, 2, 1)

print(f'total = {ga_TE_feature_trans.shape}')

# %%
n_subdims = 25
gds_subdims = 350
sumacc = 0
probe_list = []
num = 0
df = pd.read_csv(
    f"./cnnfeature/silhouette/cnnmodel/2km/ts2km.csv", header=None)
df_array = np.array(df)
add = int(df_array.shape[1]/sub_num)
for i in range(sub_num):
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/2km/ts2km.csv", header=None, usecols=[x for x in range(num, num+add)])
    probe_list.append(df)
    num += add
probe_array = np.array(probe_list)
# シフト量の計算
s = (probe_array.shape[2] - m) / (TE_num - 1)  # シフト量s
pr_TE_feature = []
for i in range(sub_num):
    TE_feature = []
    start = 0
    last = m
    for j in range(TE_num):
        rand = [random.randint(int(start), int(last))
                for k in range(sample_num)]
        rand.sort()
        cnnfeature = []
        for k in rand:
            x = probe_array[i][:, k-1]
            cnnfeature.extend(x)
        TE_feature.append(cnnfeature)
        # if(j % 2 == 0):
        #     ceil = math.ceil(s)
        #     start += ceil
        #     last += ceil
        # else:
        #     floor = math.floor(s)
        #     start += floor
        #     last += floor
        # if(last > probe_array.shape[2]):
        #     last = probe_array.shape[2]
        start += s
        last += s
    TE_feature = np.array(TE_feature)
    pr_TE_feature.append(TE_feature.T)
pr_TE_feature = np.array(pr_TE_feature)
pr_TE_feature_trans = pr_TE_feature.transpose(0, 2, 1)
model = ConstrainedMSM(n_subdims=n_subdims, n_gds_dims=gds_subdims)
model.fit(ga_TE_feature_trans, y)
model.n_subdims = 35
pred = model.predict(pr_TE_feature_trans)
print(f"pred: {pred}\n true: {y}\n")
accuracy = (pred == y).mean()
print(f"accuracy:{accuracy}")
