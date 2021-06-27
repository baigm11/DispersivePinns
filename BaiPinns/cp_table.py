import os
import pandas as pd

# base_path_list = ["kdv_single_cp", "kdv_double_cp", "Kawa_single_cp",
#                   "CH_single_lim1_cp", "CH_double_lim1_cp", "BO_single_enh4_1_cp", "BO_double_enh30_cp"]
# base_path_list = ["Kawa_double2_cp", "Kawa_gen_cp", "Kawa_agen_cp"]
base_path_list = ["BO_double_enh30_cp"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    directories_iter = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(directories_iter)

    info_model_total = None
    for dir_iter in directories_iter:
        #         assert os.path.isfile(base_path + "/" + dir_iter + "/InfoModel.txt") == True
        if os.path.isfile(base_path + "/" + dir_iter + "/InfoModel.txt") == True:
            info_model = pd.read_csv(base_path + "/" + dir_iter + "/InfoModel.txt", header=0, sep=",")
            info_model_total = pd.concat([info_model_total, info_model], 0)
    #         print(info_model)
    print(info_model_total)
    info_model_total.shape

df = info_model_total[['iterations', 'train_time', 'error_train', 'rel_L2_norm']].sort_values('iterations')
df = df.sort_values('iterations') # 'train_time'
with pd.option_context('display.precision', 2):
    print(df)