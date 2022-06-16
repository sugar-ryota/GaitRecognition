# from _typeshed import SupportsAnext

from RTW_limited_randomsampling_def import limited_predict

# for i in range(2, 11):
#   for j in range(2, 11):
#     predict(i, j)
TE_num = 320  # 変数R(TE特徴の数)
sample_num = 5  # ランダムサンプリングする数
save_path = "./accuracy/RTW_limited_randomsampling/RTW_limited_randomsampling14.txt"

with open(save_path, mode='a') as file:
    file.write("\n")
    file.write(f"TE_num = {TE_num}, sample_num = {sample_num}\n")
    file.write("\n")
for i in range(2, 11):
    for j in range(2, 11):
        if (abs(i - j)) <= 2:
            accuracy = limited_predict(i, j, TE_num, sample_num)
            with open(save_path, mode='a') as file:
                file.write(f"gallery:{i}km probe:{j}km")
                file.write(f"accuracy:{accuracy}")
                file.write("\n")
# predict(2,4)

# for te in TE_num:
#     for sample in samples:
#         for subdim in n_subdims:
#             gds = (subdim * 34) - subdim
#             predict(te, sample, subdim, gds)
