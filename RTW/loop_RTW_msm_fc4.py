# from _typeshed import SupportsAnext
from RTW_msm_fc4_def import predict


# for i in range(2,11):
#   for j in range(2,11):
#     predict(i,j)

for i in range(2, 11):
    for j in range(2, 11):
        if (abs(i - j)) <= 2:
            predict(i, j)

# for te in TE_num:
#     for sample in samples:
#         for subdim in n_subdims:
#             gds = (subdim * 34) - subdim
#             predict(te, sample, subdim, gds)
