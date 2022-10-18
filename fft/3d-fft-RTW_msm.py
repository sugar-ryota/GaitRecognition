#!/usr/bin/env python
# coding: utf-8

# %%
import numpy as np
import copy
import glob
from PIL import Image
from keras.utils.np_utils import to_categorical
from collections import OrderedDict
from natsort import natsorted
from numpy.fft import fftn
import random
from base.base_class import MSMInterface,SMBase
from base.base import subspace_bases


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

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# クラス数
sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)


# 順番が保持された辞書
dict = OrderedDict({
    '2km': [],
    '3km': [],
    '4km': [],
    '5km': [],
    '6km': [],
    '7km': [],
    '8km': [],
    '9km': [],
    '10km': [],
})

# dictを異なるIDでコピーする(dictが変更されても影響は受けない)
gxdata = copy.deepcopy(dict)
gydata = copy.deepcopy(dict)
pxdata = copy.deepcopy(dict)
pydata = copy.deepcopy(dict)

# naturalsort #["00001","00002"....]
sblist = natsorted(glob.glob("data/TreadmillDatasetA/*"))
for sbi, sb in enumerate(sblist):  # index object
    # リスト型 ["gallery_2km","gallery_3km"....]
    galist = natsorted(glob.glob(sb+"/gallery*"))
    prlist = natsorted(glob.glob(sb+"/probe*"))

    # gallery読み込み
    for kmi, km in enumerate(galist, 2):  # 2kmからだからindexの開始数値は2
        key = str(kmi)+'km'
        # ["00000001.png","00000002.png"....]
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic).resize((32,22)))
            tmp = tmp.reshape(32,22)
            tmp = tmp/255  # 画像を正規化
            gxdata[key].append(tmp)
            gydata[key].append(sbi)  # label

    # probe読み込み
    # 3つにわける
    for kmi, km in enumerate(prlist, 2):
        key = str(kmi)+'km'
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic).resize((32,22)))
            tmp = tmp.reshape(32,22)
            tmp = tmp/255  # 画像を正規化
            pxdata[key].append(tmp)
            pydata[key].append(sbi)  # label

# %%
glabel = copy.deepcopy(gydata)
plabel = copy.deepcopy(pydata)

# %%


for km in gxdata.keys():  # ["2km","3km",....]
    gxdata[km] = np.array(gxdata[km])
    gxdata[km].astype('float32')
    gydata[km] = np.array(gydata[km])
    gydata[km] = to_categorical(gydata[km])

# %%


for km in pxdata.keys():  # ["2km","3km",....]
    pxdata[km] = np.array(pxdata[km])
    pxdata[km].astype('float32')
    pydata[km] = np.array(pydata[km])
    pydata[km] = to_categorical(pydata[km])


TE_num = 150  # 変数R(TE特徴の数)
sample_num = 5  # ランダムサンプリングする数
gallery = 2
probe = 3
# galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
num = 0
gallery = str(gallery)+'km'
probe = str(probe)+'km'
df_array = gxdata[gallery]
print(f'df_array_shape = {df_array.shape}') #(34*学習データ数,32,22)
add = int(df_array.shape[0]/sub_num)
for i in range(sub_num):
    array = df_array[num:num+add]
    gallery_list.append(array)
    num += add
gallery_array = np.array(gallery_list) # ここにfftの処理を書く
# gallery_fft = []
ga_TE_feature = []
for subject in gallery_array:
    TE_feature = []
    for i in range(TE_num):
        rand = [random.randint(0, add-1) for s in range(sample_num)]
        rand.sort()
        fft_array = []
        for k in rand:
            x = subject[k]
            fft_array.append(x)
        #FFTの処理 fft_array.shape = (sample_num,32,22)
        fft_array = abs(fftn(fft_array))
        fft_array_flatten = fft_array.flatten()
        TE_feature.append(fft_array_flatten)
    TE_feature = np.array(TE_feature)
    # print(f'TE_feature = {TE_feature.shape}')
    ga_TE_feature.append(TE_feature)
ga_TE_feature = np.array(ga_TE_feature)
print(f'ga_TE_feature.shape = {ga_TE_feature.shape}')

probe_list = []
num = 0
df_array = pxdata[probe]
add = int(df_array.shape[0]/sub_num)
for i in range(sub_num):
    array = df_array[num:num+add]
    probe_list.append(array)
    num += add
probe_array = np.array(probe_list)
# probe_fft = []

pr_TE_feature = []
for subject in probe_array:
    TE_feature = []
    for i in range(TE_num):
        rand = [random.randint(0, add-1) for s in range(sample_num)]
        rand.sort()
        fft_array = []
        for k in rand:
            x = subject[k]
            fft_array.append(x)
        #FFTの処理 fft_array.shape = (sample_num,32,22)
        fft_array = abs(fftn(fft_array))
        fft_array_flatten = fft_array.flatten()
        TE_feature.append(fft_array_flatten)
    TE_feature = np.array(TE_feature) #(TE_num,sample_num*32*22)
    pr_TE_feature.append(TE_feature)
pr_TE_feature = np.array(pr_TE_feature)
print(f'pr_TE_feature.shape = {pr_TE_feature.shape}')


pred = []
model = MutualSubspaceMethod(n_subdims=50)
model.fit(ga_TE_feature, y)
model.n_subdims = 50
pred,proba = model.predict(pr_TE_feature)
print(f"pred: {pred}\n true: {y}\n")
accuracy = (pred == y).mean()
print(f"accuracy:{accuracy}")
