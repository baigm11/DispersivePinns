from CollectUtils import *

np.random.seed(42)


base_path_list = ["kdv_double"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    compute_std = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["batch_size",
                                           "regularization_parameter",
                                           "kernel_regularizer",
                                           "neurons",
                                           "hidden_layers",
                                           "residual_parameter",
                                           "L2_norm_test",
                                           "error_train",
                                           "error_val",
                                           "error_test"])
    # print(sensitivity_df)
    selection_criterion = "metric_2"

    for subdirec in directories_model:
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            if os.path.isfile(sample_path + "/" + ret + "/TrainedModel/Information.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "/" + retr_to_check_file + "/TrainedModel/Information.csv", header=0, sep=",")
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="min", compute_std=compute_std, compute_val=False, rs_val=0)
            best_retrain["setup"] = setup_num
            best_retrain = best_retrain.to_frame()
            best_retrain = best_retrain.transpose().reset_index().drop("index", 1)
            info_model = pd.concat([info_model, best_retrain], 1)

            print(info_model)

            if info_model["batch_size"].values[0] == "full":
                info_model["batch_size"] = info_model["Nu_train"] + info_model["Nf_train"]
            sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "/TrainedModel/Information.csv not found")

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    best_setup = sensitivity_df.iloc[0]
    best_setup.to_csv(base_path + "best.csv", header=0, index=False)
    # print(sensitivity_df)
    print("Best Setup:", best_setup["setup"])
    print(best_setup)

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df["rel_L2_norm"])
    plt.xlabel(r'Metric')
    plt.ylabel(r'$\varepsilon_G$')
    # plt.show()
    plt.savefig(base_path + "/et_vs_eg.png", dpi=400)

quit()
