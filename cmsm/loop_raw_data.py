# from _typeshed import SupportsAnext

from raw_data_cmsm_def import cmsm_predict
from raw_data_RTW_def import RTW_predict

# for i in range(2, 11):
#   for j in range(2, 11):
#     predict(i, j)
TE_num = 320  # 変数R(TE特徴の数)
sample_num = 5  # ランダムサンプリングする数
cmsm_save_path = "./accuracy/raw_data/cmsm_1.txt"
RTW_save_path = "./accuracy/raw_data/RTW_1.txt"

with open(RTW_save_path, mode='a') as file:
    file.write("\n")
    file.write(f"TE_num = {TE_num}, sample_num = {sample_num}\n")
    file.write("\n")
for i in range(2, 11):
    for j in range(2, 11):
        if (abs(i - j)) <= 2:
            accuracy = cmsm_predict(i,j)
            with open(cmsm_save_path, mode='a') as file:
                file.write("\n")
                file.write(f"gallery:{i}km probe:{j}km")
                file.write(f"accuracy:{accuracy}")
                file.write("\n")
            accuracy = RTW_predict(i, j, TE_num, sample_num)
            with open(RTW_save_path, mode='a') as file:
                file.write("\n")
                file.write(f"gallery:{i}km probe:{j}km")
                file.write(f"accuracy:{accuracy}")
                file.write("\n")
