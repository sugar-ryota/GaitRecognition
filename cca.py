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
# from base.base_class_cca import MSMInterface, SMBase
# from base.base_cca import subspace_bases
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
import cv2
import matplotlib.pyplot as plt


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
            tmp = np.array(Image.open(pic)).reshape(-1)
            tmp = tmp/255  # 画像を正規化
            gxdata[key].append(tmp)
            gydata[key].append(sbi)  # label

    # probe読み込み
    # 3つにわける
    for kmi, km in enumerate(prlist, 2):
        key = str(kmi)+'km'
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic)).reshape(-1)
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
probe = 3
# galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
gallery_list = []
num = 0
gallery = str(gallery)+'km'
probe = str(probe)+'km'
df_array = gxdata[gallery]
add = int(df_array.shape[0]/sub_num)
for i in range(sub_num):
    array = df_array[num:num+add]
    gallery_list.append(array)
    num += add
gallery_array = np.array(gallery_list)

probe_list = []
num = 0
df_array = pxdata[probe]
add = int(df_array.shape[0]/sub_num)
for i in range(sub_num):
    array = df_array[num:num+add]
    probe_list.append(array)
    num += add
probe_array = np.array(probe_list)

# 一番目の被験者同士を比べる
first_gallery_array = gallery_array[0][0]
first_gallery_array = first_gallery_array.reshape(1, 11264)
first_probe_array = probe_array[0]
print(f'train_shape = {first_gallery_array.shape}')
print(f'test_shape = {first_probe_array.shape}')  # test_shape=(420,11264)

# s_xx = (first_gallery_array @ first_gallery_array.T) / \
#     first_gallery_array.shape[0]
s_xx = (first_gallery_array @ first_gallery_array.T)
s_yy = (first_probe_array @ first_probe_array.T) / first_probe_array.shape[0]
s_xy = (first_gallery_array @ first_probe_array.T) / \
    first_gallery_array.shape[0]
s_yx = (first_probe_array @ first_gallery_array.T) / first_probe_array.shape[0]

s_a = np.linalg.inv(s_xx)@s_xy@np.linalg.inv(s_yy)@s_yx
s_b = np.linalg.inv(s_yy)@s_yx@np.linalg.inv(s_xx)@s_xy

print(f's_a_shape = {s_a.shape}')
print(f's_b_shape = {s_b.shape}')

s_a_eighs = np.linalg.eig(s_a)[0]
s_b_eighs = np.linalg.eig(s_b)[0]

s_a_eighvec = np.linalg.eig(s_a)[1]
s_b_eighvec = np.linalg.eig(s_b)[1]


# save_path = "./cca2_3.txt"
# with open(save_path, mode='a') as file:
#     file.write(f"a_eighs:{s_a_eighs}")
#     file.write("\n")
#     file.write(f"a_vectors:{s_a_eighvec}")
#     file.write("\n")
#     file.write(f"b_eighs:{s_b_eighs}")
#     file.write("\n")
#     file.write(f"b_vectors:{s_b_eighs}")
#     file.write("\n")
print(f's_a_eigh_shape = {s_a_eighs.shape}')
print(f's_a_eigh_vector_shape = {s_a_eighvec.shape}')
print(f's_b_eigh_shape = {s_b_eighs.shape}')
print(f's_b_eigh_vector_shape = {s_b_eighvec.shape}')
# a_max = np.max(s_a_eighs)
# a_max_index = np.where(s_a_eighs == a_max)
# a_max_vector = s_a_eighvec[a_max_index][:]  # aの最大固有値に対する固有ベクトル
# print(a_max_vector.shape)
b_max = np.max(s_b_eighs)
b_max_index = np.where(s_b_eighs == b_max)
b_max_vector = s_b_eighvec[b_max_index][:]  # bの最大固有値に対する固有ベクトル
# a_vec_max = np.max(a_max_vector)
# b_vec_max = np.max(b_max_vector)
# a_vec_max_index = np.where(a_max_vector == a_vec_max)
# b_vec_max_index = np.where(b_max_vector == b_vec_max)
b_max_vector = np.array(b_max_vector)
# print(b_max_vector.shape)
result_array = np.argsort(b_max_vector)
# print(result_array)


# print(f'eighs = {eighs}')
# print(f'eigh_vectors = {eigh_vectors}')
# value, vector = eig(first_gallery_array.T, first_probe_array.T)
# value = np.array(value)
# vector = np.array(vector)
# print(f'value ={value.shape}')
# print(f'vector ={vector.shape}')

# 画像の描画
filled_array = []
for number in result_array[0]:
    filled_number = str(number+1).zfill(8)
    filled_array.append(filled_number)
print(filled_array)
n_data = 12
row = 3
col = 4
fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8, 6))
for i in range(n_data):
    r = i//col
    c = i % col
    img = Image.open(
        f"data/TreadmillDatasetA/00001/gallery_3km/{filled_array[-i]}.png")
    ax[r, c].axes.xaxis.set_visible(False)  # X軸を非表示に
    ax[r, c].axes.yaxis.set_visible(False)  # Y軸を非表示に
    ax[r, c].set_title(filled_array[-i-1])
    # fig.savefig('result.jpg', cmap="gray")
    ax[r, c].imshow(img, cmap='gray')  # 画像を表示
fig.savefig('last.png')
# 保存する
# plt.imsave("sample.png", img, cmap="gray")

# img = Image.open(
#     f"data/TreadmillDatasetA/00001/gallery_3km/" + "00000001.png")
# plt.imsave("sample.png", img, cmap="gray")
