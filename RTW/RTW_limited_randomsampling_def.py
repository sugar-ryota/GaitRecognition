# 目視で周期をきめつつ、ランダムサンプリングをする際に制限をつける
# 具体的には1周期の中で、最初と最後の部分は必ず取得するようにしてみる

# %%

import numpy as np
import pandas as pd
import random

from base.base_class_log import ConstrainedSMBase, MSMInterface
from base.base_log import subspace_bases

# %%


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
# %%


sub_num = 34
y = []
for i in range(sub_num):
    y.append(i)

# %%
# 目視で判断した1周期分のフレーム数
mokushi_num = {"2km": 49, "3km": 41, "4km": 38, "5km": 36,
               "6km": 33, "7km": 28, "8km": 27, "9km": 22, "10km": 21}

# %%


def limited_predict(gallery, probe, TE_num, sample_num):
    gallery_list = []
    num = 0
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/{gallery}km/tr_ex.csv", header=None)
    df_array = np.array(df)
    add = int(df_array.shape[1]/sub_num)
    for i in range(sub_num):
        df = pd.read_csv(f"./cnnfeature/silhouette/cnnmodel/{gallery}km/tr_ex.csv",
                         header=None, usecols=[x for x in range(num, num+add)])
        gallery_list.append(df)
        num += add
    gallery_array = np.array(gallery_list)
    # %%
    repetition_num = 8
    # 目視で判断した1周期分だけ取り出す
    extract_num_train = mokushi_num[str(gallery) + 'km']

    ga_TE_feature = []
    for i in range(sub_num):
        TE_feature = []
        start = 0
        end = extract_num_train
        for j in range(repetition_num):
            for l in range(int(TE_num/repetition_num)):
                # ランダムサンプリングに制限をつける
                cnnfeature = []
                # cnnfeature.extend(gallery_array[i][:, start])
                rand = [random.randint(start+2, end-2)
                        for k in range(sample_num)]
                rand.sort()
                for k in rand:
                    x = gallery_array[i][:, k-1]
                    cnnfeature.extend(x)
                # cnnfeature.extend(gallery_array[i][:, end])
                TE_feature.append(cnnfeature)
            start = end
            end += extract_num_train
        TE_feature = np.array(TE_feature)
        ga_TE_feature.append(TE_feature.T)
    ga_TE_feature = np.array(ga_TE_feature)
    ga_TE_feature_trans = ga_TE_feature.transpose(0, 2, 1)

    # %%
    n_subdims = 25
    gds_subdims = 350
    probe_list = []
    num = 0
    df = pd.read_csv(
        f"./cnnfeature/silhouette/cnnmodel/{gallery}km/ts{probe}km.csv", header=None)
    df_array = np.array(df)
    add = int(df_array.shape[1]/sub_num)
    for i in range(sub_num):
        df = pd.read_csv(
            f"./cnnfeature/silhouette/cnnmodel/{gallery}km/ts{probe}km.csv", header=None, usecols=[x for x in range(num, num+add)])
        probe_list.append(df)
        num += add
    probe_array = np.array(probe_list)

    extract_num_test = mokushi_num[str(probe) + 'km']

    pr_TE_feature = []
    for i in range(sub_num):
        TE_feature = []
        start = 0
        end = extract_num_test
        for j in range(repetition_num):
            for l in range(int(TE_num/repetition_num)):
                cnnfeature = []
                # cnnfeature.extend(probe_array[i][:, start])
                rand = [random.randint(start+2, end-2)
                        for k in range(sample_num)]
                rand.sort()
                for k in rand:
                    x = probe_array[i][:, k-1]
                    cnnfeature.extend(x)
                # cnnfeature.extend(probe_array[i][:, end])
                TE_feature.append(cnnfeature)
            start = end
            end += extract_num_test
        TE_feature = np.array(TE_feature)
        pr_TE_feature.append(TE_feature.T)
    pr_TE_feature = np.array(pr_TE_feature)
    pr_TE_feature_trans = pr_TE_feature.transpose(0, 2, 1)
    model = ConstrainedMSM(n_subdims=n_subdims, n_gds_dims=gds_subdims)
    model.fit(ga_TE_feature_trans, y)
    model.n_subdims = 35
    pred = model.predict(pr_TE_feature_trans)
    accuracy = (pred == y).mean()
    print(f"limited_accuracy:{accuracy}")
    return accuracy
