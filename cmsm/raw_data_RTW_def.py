#!/usr/bin/env python
# coding: utf-8

# %%
import os
import numpy as np
import copy
import glob
from PIL import Image
from keras.utils.np_utils import to_categorical
import gc
from collections import OrderedDict
from natsort import natsorted
from base.base_class import ConstrainedSMBase, MSMInterface
from base.base import subspace_bases
import random 



import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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

#クラス数
sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)


#順番が保持された辞書
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

#dictを異なるIDでコピーする(dictが変更されても影響は受けない)
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

    #gallery読み込み
    for kmi, km in enumerate(galist, 2):  # 2kmからだからindexの開始数値は2
        key = str(kmi)+'km'
        # ["00000001.png","00000002.png"....]
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic)).reshape(-1)
            tmp = tmp/255  # 画像を正規化
            gxdata[key].append(tmp)
            gydata[key].append(sbi)  # label

    #probe読み込み
    #3つにわける
    for kmi, km in enumerate(prlist, 2):
        key = str(kmi)+'km'
        piclist = natsorted(glob.glob(km+"/*.png"))
        for pic in piclist:
            tmp = np.array(Image.open(pic)).reshape(-1)
            tmp = tmp/255
            pxdata[key].append(tmp)
            pydata[key].append(sbi)  # label

del sblist
gc.collect()
del piclist
gc.collect()
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

# 目視で判断した1周期分のフレーム数
mokushi_num = {"2km": 49, "3km": 41, "4km": 38, "5km": 36,
               "6km": 33, "7km": 28, "8km": 27, "9km": 22, "10km": 21}

def RTW_predict(gallery,probe,TE_num,sample_num):

    # galleryの特徴量に関して、それぞれの被験者ごとに配列に分ける
    gallery_list = []
    num = 0
    gallery = str(gallery)+'km'

    repetition_num = 8
    # 目視で判断した1周期分だけ取り出す
    extract_num_train = mokushi_num[gallery]
    df_array = gxdata[gallery]
    add = int(df_array.shape[0]/sub_num)
    start = 0
    end = extract_num_train
    for i in range(sub_num):
        TE_feature= []
        for j in range(repetition_num):
            for l in range(int(TE_num/repetition_num)):
                rand = [random.randint(start+2, end-2-1) for k in range(sample_num)]
                rand.sort()
                feature = []
                for k in rand:
                    x = df_array[k]
                    feature.extend(x)
                TE_feature.append(feature)
            start = end
            end += extract_num_train
        gallery_list.append(TE_feature)
        num += add
        start = num
        end = num + extract_num_train
    gallery_array = np.array(gallery_list)

    probe_list = []
    num = 0
    probe = str(probe)+'km'

    extract_num_test = mokushi_num[probe]
    df_array = pxdata[probe]
    add = int(df_array.shape[0]/sub_num)
    start = 0
    end = extract_num_test
    for i in range(sub_num):
        TE_feature = []
        for j in range(repetition_num):
            for l in range(int(TE_num/repetition_num)):
                rand = [random.randint(start+2, end-2-1) for k in range(sample_num)]
                rand.sort()
                feature = []
                for k in rand:
                    x = df_array[k]
                    feature.extend(x)
                TE_feature.append(feature)
            start = end
            end += extract_num_test
        probe_list.append(TE_feature)
        num += add
        start = num
        end = num + extract_num_test
    probe_array = np.array(probe_list)
    model = ConstrainedMSM(n_subdims=10, n_gds_dims=240)
    model.fit(gallery_array, y)
    model.n_subdims = 15
    pred = model.predict(probe_array)
    print(f"pred: {pred}\n true: {y}\n")
    accuracy = (pred == y).mean()
    print(f"accuracy:{accuracy}")
    return accuracy
# %%
