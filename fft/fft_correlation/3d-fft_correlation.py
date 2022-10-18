#420枚を2分割してみてその相関を調べてみる
#具体的には、2分割したのちに3d-fftをかける→一本のベクトルを生成する
#その2つの相関を調べてみる


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

first = gallery_array[0][0:210]
second = gallery_array[0][210:420]

first = np.array(first)
second = np.array(second)
# print(first.shape)
# print(second.shape)

f_fft = abs(fftn(first))
s_fft = abs(fftn(second))

print(f_fft.shape)
print(s_fft.shape)

f_flatten = f_fft.flatten()
s_flatten = s_fft.flatten()

cos = np.inner(f_flatten, s_flatten) / (np.linalg.norm(f_flatten) * np.linalg.norm(s_flatten))

print(cos)