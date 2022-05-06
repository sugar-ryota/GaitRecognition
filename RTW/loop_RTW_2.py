# from _typeshed import SupportsAnext
import numpy as np
import csv
import pandas as pd
import pickle
import random

from base.base_class import ConstrainedSMBase, MSMInterface
from base.base import subspace_bases, mean_square_singular_values

from RTW.RTW_dimention_def2 import ConstrainedMSM
from RTW.RTW_dimention_def2 import predict


for i in range(10,60,5):
    predict(15,300,i)

# for te in TE_num:
#     for sample in samples:
#         for subdim in n_subdims:
#             gds = (subdim * 34) - subdim
#             predict(te, sample, subdim, gds)
