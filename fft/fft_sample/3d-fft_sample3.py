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
km = str(2)
gxdata_array = np.array(gxdata[str(km)+'km'])
frame_num = int((gxdata_array.shape[0])/sub_num)
frame_array = []
for i in range(frame_num):
    frame_array.append(gxdata[str(km)+'km'][i])
frame_array = np.array(frame_array)

sizefft = [128, 88, frame_num]
array3d = np.array(frame_array.transpose(1, 2, 0)
                   )  # array3d.shape = 128,88,420

# w = np.hanning(frame_num) #ハニング窓

# 3d-fftをした結果が格納されている
fft_array = fftn(array3d, sizefft)

# fftした結果の画像表示
fft_array = np.fft.fftshift(fft_array)
fft_array = abs(fft_array)

fft_array = fft_array/max(fft_array.flatten())
fft_array = np.power(fft_array, 0.1)

# wt, wh, wv = np.meshgrid(
#     -1: 2/sizefft[3]: 1-2/sizefft[3],
#     -1: 2/sizefft[1]: 1-2/sizefft[1],
#     -1: 2/sizefft[2]: 1-2/sizefft[2],
# )
# fft_array = fft_array.transpose(1, 2, 0) #shape = 88,420,128

print(fft_array[0][1].shape)
print(fft_array[1].shape)
print(fft_array[2].shape)

fig, ax = plt.subplots()
ax = fig.add_subplot(projection='3d')
ax.scatter(fft_array[0], fft_array[1], fft_array[2])
ax.set_xlabel('height')
ax.set_ylabel('width')
ax.set_zlabel('time')
# # ax.set_xlim(0, 0.1)
# ax.grid()
plt.savefig(f'fft/sample3.png')
# # plt.savefig(f'sample.png')
# # plt.savefig(f'fft/plot/{km}km.png')
# # plt.show()


# plt.plot(freq[1:int(N/2)]/1000 , Amp[1:int(N/2)])
# plt.xlim(0,3)
# plt.xlabel("Frequency [kHz]")
# plt.ylabel("Amplitude [#]")
# plt.title("FFT test")


# %%
