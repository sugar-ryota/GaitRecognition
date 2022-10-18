# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import tensorflow
import copy
import glob
import openpyxl as excel
from collections import OrderedDict
import openpyxl as excel
from PIL import Image
# from keras.preprocessing.image import array_to_img
from keras.utils.np_utils import to_categorical
from pathlib import Path
from natsort import natsorted
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
# from sklearn.datasets import fetch_mldata
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from numpy.fft import fftn

# %%
# データセットを読み込んで学習データと訓練データに分ける
# そして保存するためのファイル

# クラス数
sub_num = 34

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
pxdata = []
pydata = []

for i in range(3):
    pxdata.append(copy.deepcopy(dict))
    pydata.append(copy.deepcopy(dict))

# %%
# naturalsort #["00001","00002"....]
sblist = natsorted(glob.glob("data/TreadmillDatasetA/*"))
for sbi, sb in enumerate(sblist):  # index object
    # リスト型 ["gallery_2km","gallery_3km"....]
    galist = natsorted(glob.glob(sb+"/gallery*"))
    prlist = natsorted(glob.glob(sb+"/probe*"))

    # gallery読み込み
    for kmi, km in enumerate(galist, 2):  # 2kmからだからindexの開始数値は2
        key = str(kmi)+'km'
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic)).reshape(
                128, 88)
            tmp = tmp/255  # 画像を正規化
            gxdata[key].append(tmp)
            gydata[key].append(sbi)

    # probe読み込み
    # 3つにわける
    for kmi, km in enumerate(prlist, 2):
        key = str(kmi)+'km'
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pici, pic in enumerate(piclist):
            tmp = np.array(Image.open(pic)).reshape(128, 88)
            tmp = tmp/255
            if(pici < len(piclist) / 3):
                i = 0
            elif(pici < len(piclist) / 3*2):
                i = 1
            else:
                i = 2
            pxdata[i][key].append(tmp)
            pydata[i][key].append(sbi)  # label

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
for i in range(3):
    for km in pxdata[i].keys():  # ["2km","3km",....]
        pxdata[i][km] = np.array(pxdata[i][km])
        pxdata[i][km].astype('float32')
        pydata[i][km] = np.array(pydata[i][km])
        pydata[i][km] = to_categorical(pydata[i][km])


# 画像の表示関数
def show(img):
    plt.figure()
    plt.imshow(img)
    plt.gray()
    plt.show()


# %%
km = str(2)
gxdata_array = np.array(gxdata[str(km)+'km'])
frame_num = int((gxdata_array.shape[0])/sub_num)
frame_array = []
for i in range(frame_num):
    frame_array.append(gxdata[str(km)+'km'][i])

array3d = np.array(frame_array)

# 3d-fftをした結果が格納されている
fft_array = fftn(array3d)
fft_array = np.fft.fftshift(fft_array)
fft_array = abs(fft_array)

# データ数
N = frame_num
dt = 0.03  # サンプリング周期(s) 一回のデータをサンプリングするのに何秒かかかるか

# freqは周波数を表す
freq = np.fft.fftfreq(N, d=dt)
freq_array = np.array(freq)

fft_array = fft_array.transpose(1, 2, 0)
# Ampは振幅
Amp = np.abs(fft_array[0][1]/(N/2))
print(Amp.shape)

fig, ax = plt.subplots()
ax.plot(freq_array[1:int(N/2)], Amp[1:int(N/2)])
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
ax.grid()
plt.savefig(f'fft/plot/{km}km.png')

# %%
