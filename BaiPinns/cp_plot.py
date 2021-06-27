import os
import matplotlib.pyplot as plt
import pandas as pd

# base_path_list = ["kdv_single_cp", "kdv_double_cp", "Kawa_single_cp",
#                   "CH_single_lim1_cp", "CH_double_lim1_cp", "BO_single_enh4_1_cp", "BO_double_enh30_cp"]
# base_path_list = ["Kawa_double2_cp", "Kawa_gen_cp", "Kawa_agen_cp"]
base_path_list = ["BO_double_enh30_cp"]
folder_path = "Test" # "cp_fig", "Test"

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
    df = df.sort_values('train_time')
    iterations = df['iterations'].values
    train_time = df['train_time'].values
    error_train = df['error_train'].values
    rel_L2_norm = df['rel_L2_norm'].values
    print(rel_L2_norm)

    # one y axis, for cases not having exact sol
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    #
    # ax.set_yscale('log')
    # ax.scatter(x=train_time, y=error_train, label='train error', marker="o", s=14)
    # ax.plot(train_time, error_train)
    # ax.set_xlabel('train time' + r'$/s$')
    # ax.set_ylabel('train error')
    #
    # ax.legend()

    # two y axis
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel('train time' + r'$/s$')
    ax1.set_ylabel('train error')
    ax1.set_yscale('log')
    lns1 = ax1.scatter(x=train_time, y=error_train, label='train error', marker="o", s=14)
    ax1.plot(train_time, error_train)
    ax1.legend(loc=2)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('rel L2 norm')
    ax2.set_yscale('log')
    lns2 = ax2.scatter(x=train_time, y=rel_L2_norm, label='rel L2 norm', marker="o", s=14, color='r')
    ax2.plot(train_time, rel_L2_norm, color='r')
    ax2.legend(loc=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # fig.legend(loc="upper right")
    # plt.show()


    # save
    plt.savefig(folder_path + '/' + base_path + '.png', dpi=500)