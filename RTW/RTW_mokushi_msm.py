# %%

import numpy as np
import pandas as pd
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
# cnn特徴量を読み込んでランダムサンプリングする

# gallery_key_num = {"2km": 420, "3km": 360, "4km": 360, "5km": 420,
#                    "6km": 360, "7km": 240, "8km": 240, "9km": 240, "10km": 300}
# probe_key_num = {"2km": 140, "3km": 120, "4km": 120, "5km": 140,
#                  "6km": 120, "7km": 80, "8km": 80, "9km": 80, "10km": 100}
# 目視で判断した1周期分のフレーム数
mokushi_num = {"2km": 49, "3km": 41, "4km": 38, "5km": 32,
               "6km": 30, "7km": 28, "8km": 27, "9km": 22, "10km": 21}
TE_num = 120  # 変数R(TE特徴の数)
sample_num = 15  # ランダムサンプリングする数

# %%

train_km = 8
test_km = 8

gallery_list = []
num = 0
df = pd.read_csv(
    f"./cnnfeature/silhouette/cnnmodel/{train_km}km/tr_ex.csv", header=None)
df_array = np.array(df)
add = int(df_array.shape[1]/sub_num)
for i in range(sub_num):
    df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/{train_km}km/tr_ex.csv",
                     header=None, usecols=[x for x in range(num, num+add)])
    gallery_list.append(df)
    num += add
gallery_array = np.array(gallery_list)
# %%
repetition_num = 8
# 目視で判断した1周期分だけ取り出す
extract_num_train = mokushi_num[str(train_km) + 'km']

ga_TE_feature = []
for i in range(sub_num):
    TE_feature = []
    start = 0
    end = extract_num_train
    for j in range(repetition_num):
        for l in range(int(TE_num/repetition_num)):
            rand = [random.randint(start, end) for k in range(sample_num)]
            rand.sort()
            cnnfeature = []
            for k in rand:
                x = gallery_array[i][:, k-1]
                cnnfeature.extend(x)
            TE_feature.append(cnnfeature)
        start = end
        end += extract_num_train
    TE_feature = np.array(TE_feature)
    ga_TE_feature.append(TE_feature.T)
ga_TE_feature = np.array(ga_TE_feature)
ga_TE_feature_trans = ga_TE_feature.transpose(0, 2, 1)
# %%
n_subdims = 25
sum_acc = 0
probe_list = []
num = 0
df = pd.read_csv(
    f"./cnnfeature/silhouette/cnnmodel/{train_km}km/ts{test_km}km.csv", header=None)
df_array = np.array(df)
add = int(df_array.shape[1]/sub_num)
for i in range(sub_num):
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/{train_km}km/ts{test_km}km.csv", header=None, usecols=[x for x in range(num, num+add)])
    probe_list.append(df)
    num += add
probe_array = np.array(probe_list)

extract_num_test = mokushi_num[str(test_km) + 'km']

pr_TE_feature = []
for i in range(sub_num):
    TE_feature = []
    start = 0
    end = extract_num_test
    for j in range(repetition_num):
        for l in range(int(TE_num/repetition_num)):
            rand = [random.randint(start, end) for k in range(sample_num)]
            rand.sort()
            cnnfeature = []
            for k in rand:
                x = probe_array[i][:, k-1]
                cnnfeature.extend(x)
            TE_feature.append(cnnfeature)
        start = end
        end += extract_num_test
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
print(f"accuracy:{accuracy}")
