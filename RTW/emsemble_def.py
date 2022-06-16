# from _typeshed import SupportsAnext
import numpy as np
import csv
import pandas as pd
import pickle
import random

from base.base_class import ConstrainedSMBase, MSMInterface
from base.base import subspace_bases, mean_square_singular_values

from RTW_limited_randomsampling_def import ConstrainedMSM
from RTW_limited_randomsampling_def import limited_predict
from cmsm_ex_def import predict

TE_num = 440  # 変数R(TE特徴の数)
sample_num = 20  # ランダムサンプリングする数
save_path = "./accuracy/RTW_ensemble/RTW_ensemble_20.txt"

with open(save_path, mode='a') as file:
    file.write("\n")
    file.write(f"TE_num = {TE_num}, sample_num = {sample_num}\n")
    file.write("\n")
for i in range(2, 11):
    for j in range(2, 11):
        if (abs(i - j)) <= 2:
            cmsm_accuracy = predict(i, j)
            limited_accuracy = limited_predict(i, j, TE_num, sample_num)
            sum_acc = (cmsm_accuracy + limited_accuracy) / 2
            with open(save_path, mode='a') as file:
                file.write("\n")
                file.write(f"gallery:{i}km probe:{j}km")
                file.write("\n")
                file.write(f"cmsm_accuracy:{cmsm_accuracy}")
                file.write("\n")
                file.write(f"limited_accuracy:{limited_accuracy}")
                file.write("\n")
                file.write(f"total_accuracy:{sum_acc}")
                file.write("\n")
