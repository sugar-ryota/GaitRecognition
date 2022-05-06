# %%

import numpy as np
import csv
import pandas as pd
import pickle
import random

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


sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)

# %%
#cnn特徴量を読み込んでランダムサンプリングする

# gallery_key_num = {"2km": 420, "3km": 360, "4km": 360, "5km": 420,
#                    "6km": 360, "7km": 240, "8km": 240, "9km": 240, "10km": 300}
# probe_key_num = {"2km": 140, "3km": 120, "4km": 120, "5km": 140,
#                  "6km": 120, "7km": 80, "8km": 80, "9km": 80, "10km": 100}

TE_num = 100  # 変数R(TE特徴の数)
sample_num = 5  # ランダムサンプリングする数

# %%

def predict(ga_dimention,gds_dimention,pr_dimention):
  gallery_list = []
  num = 0
  df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/2km/tr.csv", header=None)
  df_array = np.array(df)
  print(df_array.shape)
  add = int(df_array.shape[1]/sub_num)
  for i in range(sub_num):
    df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/2km/tr.csv",
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
  #%%
  # total_TE_feature = np.array(total_TE_feature)
  # print(total_TE_feature.shape)
  save_path = "./RTW_pr_dim_result.txt"

  with open(save_path, mode='a') as file:
    file.write("\n")
    file.write("\n")
    file.write(f"ga_dim:{ga_dimention} gds_dim:{gds_dimention} pr_dim:{pr_dimention}")
    file.write("\n")
  # %%
  n_subdims = ga_dimention
  gds_subdims = gds_dimention
  sum_acc = 0
  for p_num in range(1, 4):
    print(f"p_num = {p_num}")
    probe_list = []
    num = 0
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/2km/ts4km_{p_num}.csv", header=None)
    df_array = np.array(df)
    add = int(df_array.shape[1]/sub_num)
    for i in range(sub_num):
      df = pd.read_csv(
          f"./cnnfeature/silhouette/cnnmodel/2km/ts4km_{p_num}.csv", header=None, usecols=[x for x in range(num, num+add)])
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
    model = ConstrainedMSM(n_subdims=n_subdims, n_gds_dims=gds_subdims)
    model.fit(ga_TE_feature_trans, y)
    model.n_subdims = pr_dimention
    pred = model.predict(pr_TE_feature_trans)
    print(f"pred: {pred}\n true: {y}\n")
    accuracy = (pred == y).mean()
    print(f"accuracy:{accuracy}")
    sum_acc += accuracy
    acc = sum_acc/p_num
    print(f"total_accuracy:{acc}")
    with open(save_path, mode='a') as file:
      file.write(f"accuracy:{accuracy}")
      file.write(f"total_accuracy:{acc}")
      file.write("\n")
