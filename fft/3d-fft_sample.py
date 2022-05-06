# %%
import numpy as np
import matplotlib.pyplot as plt
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
        # ["00000001.png","00000002.png"....]
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic)).reshape(
                128, 88)  # 変えた((128,88,1) -> (128,88))
            tmp = tmp/255  # 画像を正規化
            gxdata[key].append(tmp)
            gydata[key].append(sbi)  # label
        # if key == '2km':
        #     gxdata_array = np.array(gxdata[key])
        #     print(f'shape={gxdata_array.shape}')

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

# gxdata_array= np.array(gxdata['2km'])
# print(gxdata_array.shape)
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
km = str(10)
gxdata_array = np.array(gxdata[str(km)+'km'])
frame_num = int((gxdata_array.shape[0])/sub_num)
frame_array = []
for i in range(frame_num):
    frame_array.append(gxdata[str(km)+'km'][i])

array = np.array(frame_array)
# print(array.shape)(420,128,88)
# show(array[0])

# 3d-fftをした結果が格納されている
fft_array = fftn(array, axes=(1, 2))

# fftした結果の画像表示
fft_array_shift = np.fft.fftshift(fft_array)
a = np.array(fft_array_shift)
print(fft_array_shift.shape)
# show(fft_array_shift[0].real)

fft_rev = np.fft.fftshift(fft_array_shift)
fft_rev = np.fft.ifftn(fft_rev)
# show(fft_rev[0].real)


N = 2**10  # データ数
dt = 0.05  # サンプリング周期(s)

# freqは周波数を表す
freq = np.fft.fftfreq(N, d=dt)
freq_array = np.array(freq)
# print(freq_array) #後半は負の周波数となる(N/2以降)

# Ampは振幅
Amp = np.abs(fft_array/(N/2))
# print(Amp)
# print(Amp.max())

Amp_flatten = Amp.flatten()
y = []
for i in range(frame_num*128*88):
    y.append(i)


plt.plot(y, Amp_flatten)
plt.savefig(f'fft/plot/{km}km.png')
# plt.show()

# 振幅の最大値のインデックスを取得する

# plt.plot(freq[1:int(N/2)]/1000 , Amp[1:int(N/2)])
# plt.xlim(0,3)
# plt.xlabel("Frequency [kHz]")
# plt.ylabel("Amplitude [#]")
# plt.title("FFT test")

idx = np.unravel_index(np.argmax(Amp), Amp.shape)
print(f'index = {idx}')
