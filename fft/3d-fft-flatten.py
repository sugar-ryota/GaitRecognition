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
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

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
            tmp = np.array(Image.open(pic)).reshape(128,88)
            tmp = tmp/255  # 画像を正規化
            gxdata[key].append(tmp)
            gydata[key].append(sbi)  # label

    # probe読み込み
    # 3つにわける
    for kmi, km in enumerate(prlist, 2):
        key = str(kmi)+'km'
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic)).reshape(128,88)
            tmp = tmp/255
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


gallery = 2
probe = 2
# galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
num = 0
gallery = str(gallery)+'km'
probe = str(probe)+'km'
df_array = gxdata[gallery]
# print(f'df_array_shape = {df_array.shape}') (14280,128,88)
add = int(df_array.shape[0]/sub_num)
for i in range(sub_num):
    array = df_array[num:num+add]
    gallery_list.append(array)
    num += add
gallery_array = np.array(gallery_list)
# print(f'gallery_array_shape = {gallery_array.shape}')(34,420,128,88)
# ここにfftの処理を書く
gallery_fft = []
for subject in gallery_array:
    fft_subject = abs(fftn(subject))
    # print(f'fft_subject_shape = {fft_subject.shape}')(420,128,88)
    fft_subject = fft_subject.flatten()
    gallery_fft.append(fft_subject)
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
    fft_subject = abs(fftn(subject))
    fft_subject = fft_subject.flatten()
    probe_fft.append(fft_subject)
probe_fft = np.array(probe_fft)

pred = []
results = []
for i in range(len(probe_fft)):
  result = []
  for j in range(len(gallery_fft)):
    #cos類似度を求める
    sim = cos_sim(probe_fft[i],gallery_fft[j])
    result.append(sim)
    # if j == 0:
    #     print(sim)
  results.append(result)
  pred.append(np.argmax(result))

print(type(results))
results = np.array(results)
print(results.shape)

# y_one_hot = label_binarize(y, classes=y)
# # print(y_one_hot)
# for i in range(len(results)):
#     fpr, tpr, thresholds = roc_curve(y_one_hot[:,i], results[:,i])
#     plt.plot(fpr, tpr, marker='o',label=f'class: {i}')
#     plt.xlabel('FPR: False positive rate')
#     plt.ylabel('TPR: True positive rate')
#     plt.grid()
#     # plt.savefig(f'plot/msm/sklearn_roc_curve_{i}.png')
# plt.legend()
# plt.savefig(f'plot/3d-fft/flatten/sklearn_roc_curve_{gallery}_{probe}.png')

print(f"pred: {pred}\n true: {y}\n")
# accuracy = (pred == y).mean()
count = 0
for i in range(len(pred)):
    if pred[i] == y[i]:
        count += 1
accuracy = count / sub_num
print(f"accuracy:{accuracy}")
