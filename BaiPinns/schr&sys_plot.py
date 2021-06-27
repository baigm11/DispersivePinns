import torch
from ImportFile import *

import os
cwd = os.getcwd()
cwd

# /Schr_double_1/Setup_12/Retrain_1
# /Schr_sech/Setup_3/Retrain_1
# /Schr_single_1/Setup_13/Retrain_2
# /Schr_single/Setup_15/Retrain_2
# /kdv_sys_single_1/Setup_7/Retrain_2
# /kdv_sys_double_1/Setup_8/Retrain_4
path = cwd + '/kdv_sys_double_1/Setup_8/Retrain_4/TrainedModel/model.pkl'
path

model = torch.load(path)
model

images_path = cwd + "/TestBai"
Ec.plotting(model, images_path, Ec.extrema_values, None)

# Ec.compute_generalization_error(model, Ec.extrema_values, None)