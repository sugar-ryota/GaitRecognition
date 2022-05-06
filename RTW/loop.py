# from _typeshed import SupportsAnext
import numpy as np
import csv
import pandas as pd
import pickle
import random

from base.base_class import ConstrainedSMBase, MSMInterface
from base.base import subspace_bases, mean_square_singular_values

from RTW_cmsm_def import ConstrainedMSM
from RTW_cmsm_def import predict

# ここに試したい数を入れる
TE_num = [100, 200, 400,800]
samples = [30,50,70,100]
n_subdims = [5, 6, 8, 10]
# n_gds_num = [45]

sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)

  #cnn特徴量を読み込んでランダムサンプリングする

  # gallery_key_num = {"2km": 420, "3km": 360, "4km": 360, "5km": 420,
  #                    "6km": 360, "7km": 240, "8km": 240, "9km": 240, "10km": 300}
  # probe_key_num = {"2km": 140, "3km": 120, "4km": 120, "5km": 140,
  #                  "6km": 120, "7km": 80, "8km": 80, "9km": 80, "10km": 100}

TE_num = te  # 変数R(TE特徴の数)
sample_num = sample  # ランダムサンプリングする数

  #galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
num = 0
df = pd.read_csv("./cnnfeature/silhouette/cnnmodel/3km/tr.csv", header=None)
df_array = np.array(df)
print(df_array.shape)
add = int(df_array.shape[1]/sub_num)
for i in range(sub_num):
    df = pd.read_csv("./cnnfeature/silhouette/cnnmodel/3km/tr.csv",header=None, usecols=[x for x in range(num, num+add)])
    gallery_list.append(df)
    num += add
gallery_array = np.array(gallery_list)
# gallery_trans = gallery_array.transpose(0, 2, 1)

for te in TE_num:
    for sample in samples:
        for subdim in n_subdims:
            gds = (subdim * 34) - subdim
            predict(te, sample, subdim, gds)

