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

def predict(sampling_num,start,end,dic_dim,in_dim):
    gallery = 2
    probe = 2
    sampling_num = sampling_num
    steps = np.linspace(start,end,150)
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
    gallery_array = np.array(gallery_list)
    print(f'gallery_array_shape = {gallery_array.shape}') #(34,420,32,22)
    # ここにfftの処理を書く
    gallery_fft = []
    for subject in gallery_array:
        subject_array = []
        for step in steps:
            moving_array = []
            count = 0
            frame_num = float(0.0)
            while count < sampling_num:
                if frame_num.is_integer():
                    moving_array.append(subject[int(frame_num)])
                else:
                    img = subject[int(frame_num)]*float(frame_num - int(frame_num)) + subject[int(frame_num)+ 1]*float(int(frame_num) - frame_num + 1)
                    moving_array.append(img)
                count += 1
                frame_num += float(step)
                # print(f'c = {count}')
                # print(f'f = {frame_num}')
            fft_subject = abs(fftn(moving_array))
            fft_subject = fft_subject.flatten()
            subject_array.append(fft_subject)
        gallery_fft.append(subject_array)

    gallery_fft = np.array(gallery_fft)
    print(f'gallery_fft.shape = {gallery_fft.shape}')

    probe_list = []
    num = 0
    df_array = pxdata[probe]
    add = int(df_array.shape[0]/sub_num)
    for i in range(sub_num):
        array = df_array[num:num+add]
        probe_list.append(array)
        num += add
    probe_array = np.array(probe_list)
    probe_fft = []
    for subject in probe_array:
        subject_array = []
        for step in steps:
            moving_array = []
            count = 0
            frame_num = float(0.0)
            while count < sampling_num:
                if frame_num.is_integer():
                    moving_array.append(subject[int(frame_num)])
                else:
                    img = subject[int(frame_num)]*float(frame_num - int(frame_num)) + subject[int(frame_num)+ 1]*float(int(frame_num) - frame_num + 1)
                    moving_array.append(img)
                count += 1
                frame_num += float(step)
            fft_subject = abs(fftn(moving_array))
            fft_subject = fft_subject.flatten()
            subject_array.append(fft_subject)
        probe_fft.append(subject_array)

    probe_fft = np.array(probe_fft)
    print(f'probe_fft.shape = {probe_fft.shape}')

    pred = []
    model = MutualSubspaceMethod(n_subdims=dic_dim)
    model.fit(gallery_fft, y)
    model.n_subdims = in_dim
    pred,proba = model.predict(probe_fft)
    print(f"pred: {pred}\n true: {y}\n")
    accuracy = (pred == y).mean()
    print(f"accuracy:{accuracy}")
    save_path = "./fft/result/3d-fft_initial_sampling_msm/accuracy2_2_2.txt"
    with open(save_path, mode='a') as file:
        file.write("\n")
        file.write(f"sampling_num = {sampling_num}, start = {start}, end = {end}, dic_dim = {dic_dim}, in_dim = {in_dim}\n")
        file.write("\n")
        file.write(f"accuracy:{accuracy}")
        file.write("\n")
