from ImportFile import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def select_over_retrainings(folder_path, selection="error_train", mode="min", compute_std=False, compute_val=False, rs_val=0):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        # print("Looking for ", retraining)
        rs = int(folder_path.split("_")[-1])
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]

        if os.path.isfile(retrain_path + "/InfoModel.txt"):
            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=0, sep=",")
            models["retraining"] = number_of_ret
            models["metric_0"] = models["error_train"]
            models["metric_1"] = models["error_wreg"]
            models["metric_2"] = models["error_res"] + models["error_vars"]

            if os.path.isfile(retrain_path + "/TrainedModel/model.pkl"):
                trained_model = torch.load(retrain_path + "/TrainedModel/model.pkl", map_location=torch.device('cpu'))
                trained_model.eval()
                # print(models)

            models_list.append(models)
            # print(models)

        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    retraining_prop = retraining_prop.sort_values(selection)
    # print("#############################################")
    # print(retraining_prop)
    # print("#############################################")
    # quit()
    if mode == "min":
        # print("#############################################")
        # print(retraining_prop.iloc[0])
        # print("#############################################")
        return retraining_prop.iloc[0]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        # print("#############################################")
        # print(retraining_prop.mean())
        # print("#############################################")
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop

