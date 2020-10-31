from CollectUtils import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.random.seed(42)
base_path_list = ["HeatH1_50_uni/", "HeatH1_100_uni/", "HeatH1_200_uni/"]
base_path_list = ["WaveH1_30_uni/", "WaveH1_60_uni/", "WaveH1_90_uni/", "WaveH1_120_uni/",
                  "WaveH1_30_uni2/", "WaveH1_60_uni2/", "WaveH1_90_uni2/", "WaveH1_120_uni2/"]
for base_path in base_path_list:
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["Nu_train",
                                           "Nf_train",
                                           "validation_size",
                                           "L2_norm_train",
                                           "L2_norm_test",
                                           "error_train",
                                           "error_val",
                                           "error_test",
                                           "sigma_u",
                                           "sigma_u_star",
                                           "sigma_res"])


    selection_criterion = "error_train"

    Nu_list = []
    Nf_list = []
    Nint_list = []
    t_0 = 0.0
    t_f = 1.0
    x_0 = -1.0
    x_f = 1.0

    extrema_values = np.array([[t_0, t_f],
                                   [x_0, x_f]])

    seeds = np.random.randint(12,200,1)

    print(seeds)

    print(np.max(extrema_values, axis=1), np.min(extrema_values, axis=1))

    sensitivity_df = list()
    for direc in directories_model:
        list_df_val = list()
        for seed in seeds:
            model_path = base_path+direc
            print("\n")
            print("##########################")
            print(model_path)

            Nu = int(direc.split("_")[0])
            Nf = int(direc.split("_")[1])
            Nint = int(direc.split("_")[2])
            if Nu not in Nu_list:
                Nu_list.append(int(Nu))
            if Nf not in Nf_list:
                Nf_list.append(int(Nf))
            if Nint not in Nint_list:
                Nint_list.append(int(Nint))

            sub_directories_model = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            list_models_sample = list()
            for subdirec in sub_directories_model:
                sample_path = model_path + "/" + subdirec
                print(sample_path)
                selected_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="min", compute_std=False, compute_val=False, rs_val=seed)
                list_models_sample.append(selected_retrain)

            samples_ds = pd.concat(list_models_sample, axis=1).T
            mean_selected_retrain = samples_ds.mean()
            #mean_selected_retrain["val_gap_int"] = np.sqrt(abs(mean_selected_retrain["res_loss"]**2 - mean_selected_retrain["res_loss_val"]**2))
            #mean_selected_retrain["val_gap_u0"] = np.sqrt(abs(mean_selected_retrain["loss_u0_train"]**2 - mean_selected_retrain["loss_u0_val"]**2))
            #mean_selected_retrain["val_gap_ub0"] = np.sqrt(abs(mean_selected_retrain["loss_ub0_train"]**2 - mean_selected_retrain["loss_ub0_val"]**2))
            #mean_selected_retrain["val_gap_ub1"] = np.sqrt(abs(mean_selected_retrain["loss_ub1_train"]**2 - mean_selected_retrain["loss_ub1_val"]**2))

            print(mean_selected_retrain)

            list_df_val.append(mean_selected_retrain)
        df_res = pd.concat(list_df_val, axis=1).T
        mean_over_val_df = df_res.mean()
        print(mean_over_val_df)
        sensitivity_df.append(mean_over_val_df)

    sensitivity_df = pd.concat(sensitivity_df, axis=1).T
    print(sensitivity_df)

    #quit()
    #sensitivity_df.to_csv("ConvergenceAnalysis.csv", index=False)



