import os
import sys
import itertools
import subprocess
from ImportFile import *

import csv

# change for different cases!!!
# "kdv_single" "kdv_double" "Kawa_single" "Kawa_double2" "Kawa_gen"
# "CH_single_lim1" "CH_double_lim1"
# "BO_single_enh4_1" "BO_double_enh30"
best_case = "Kawa_double2"
best_path = "best/" + best_case + "best.csv"
with open(best_path, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


rs = 0
N_coll = int(data[15][0])
N_u = int(data[14][0])
N_int = int(data[16][0])
n_object = 0
ob = "None"
folder_name = sys.argv[1]
point = "sobol"
validation_size = 0.0
network_properties = {
    "hidden_layers": [int(data[4][0])],  # [4, 8, 12]
    "neurons": [int(data[3][0])],  # [20, 24, 28, 32]
    "residual_parameter": [float(data[5][0])],  # [0.1, 1, 10]
    "kernel_regularizer": [2],
    "regularization_parameter": [0],
    "batch_size": [(N_coll + N_u + N_int)],
    "epochs": [1],
    "max_iter": [100, 10000], # [100, 500, 1000, 2000, 5000, 10000]
    "activation": ["tanh"],
    "optimizer": ["LBFGS"]
}
shuffle = "false"
cluster = sys.argv[2] #true True
GPU = "GeForceGTX1080Ti"  # GPU=None    # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"
n_retrain = int(data[25][0]) # num of best training config, [0, 4]

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*network_properties.values()))

i = 0
for setup in settings:
    print(setup)

    folder_path = folder_name
    print("###################################")
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "residual_parameter": setup[2],
        "kernel_regularizer": setup[3],
        "regularization_parameter": setup[4],
        "batch_size": setup[5],
        "epochs": setup[6],
        "max_iter": setup[7],
        "activation": setup[8],
        "optimizer": setup[9]
    }

    arguments = list()
    arguments.append(str(rs))
    arguments.append(str(N_coll))
    arguments.append(str(N_u))
    arguments.append(str(N_int))
    arguments.append(str(n_object))
    arguments.append(str(ob))
    arguments.append(str(folder_path))
    arguments.append(str(point))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(setup_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(setup_properties).replace("\'", "\""))
    arguments.append(str(shuffle))
    arguments.append(str(cluster))
    arguments.append(str(GPU))
    arguments.append(str(n_retrain))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub python3 single_retraining_cp.py "
        else:
            string_to_exec = "python3 single_retraining_cp.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
