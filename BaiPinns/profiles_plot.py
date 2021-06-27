import torch
from ImportFile import *
import itertools

import os
cwd = os.getcwd()
cwd


def plotting(models, iters, images_path, extrema, name):
    '''
    Function to make plots
    Args:
        model: neural network approximating the solution
        images_path: path where to save plots
        extrema:  extrema of the domain
        solid:
    '''

    for model in models:
        model.cpu()
        model = model.eval()
    n = 500
    x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], n), [n, 1])
    # time_steps = [Ec.extrema_values[0, 0], Ec.extrema_values[0, 1]] # _if
    # time_steps = [Ec.extrema_values[0, 0]] # _i
    time_steps = [Ec.extrema_values[0, 1]] # _f
    scale_vec = np.linspace(0.65, 1.55, len(time_steps))
    scale_vec_pred = np.linspace(0.65, 1.55, len(models) * len(time_steps))

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    # comment for cases not having exact sol
    for val, scale in zip(time_steps, scale_vec):
        plot_var = torch.cat([torch.tensor(()).new_full(size=(n, 1), fill_value=val), x], 1)
        plt.plot(x, Ec.exact(plot_var), 'b-', linewidth=2, label=r'Exact, $t=$' + str(val.detach().numpy()) + r'$s$',
                 color=lighten_color('grey', scale), zorder=0)

    for [val, model_ext], scale in zip(list(itertools.product(time_steps, zip(models, iters))), scale_vec_pred):
        plot_var = torch.cat([torch.tensor(()).new_full(size=(n, 1), fill_value=val), x], 1)
        #         plt.plot(x, Ec.exact(plot_var), 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=lighten_color('grey', scale), zorder=0)
        plt.scatter(plot_var[:, 1].detach().numpy(), model_ext[0](plot_var).detach().numpy(),
                    label=r'Pred, $t=$' + str(val.detach().numpy()) + r'$s$, $iters=$' + model_ext[1], marker="o", s=5,
                    color=lighten_color('C0', scale), zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend(loc=2)
    plt.savefig(images_path + "/" + name + ".png", dpi=500)

# base_path_list = ["kdv_single_cp", "kdv_double_cp", "Kawa_single_cp",
#                   "CH_single_lim1_cp", "CH_double_lim1_cp", "BO_single_enh4_1_cp", "BO_double_enh30_cp"]
# base_path_list = ["Kawa_double2_cp", "Kawa_gen_cp", "Kawa_agen_cp"]
folder = 'BO_single_enh4_1_cp'
import EquationModels.DispersiveEquation.BO as Ec
folder_path = "legend_update" # "cp_fig1"
iters = ['1000', '2000', '5000']
images_path = cwd + "/" + folder_path

models = []
for itera in iters:
    path = cwd + '/' + folder + '/iter_' + itera + '/TrainedModel/model.pkl'
    print(path)
    model = torch.load(path)
    models.append(model)

plotting(models, iters, images_path, Ec.extrema_values, folder + "_ff") # _f, _i, _ if
type(models)



# plot for /Kawa_double2/Setup_11/Retrain_2/TrainedModel/model.pkl
# path = cwd + '/Kawa_double2/Setup_11/Retrain_2/TrainedModel/model.pkl'
# path
#
# model = torch.load(path)
# model
#
#
# images_path = cwd + "/TestBai"
# Ec.plotting(model, images_path, Ec.extrema_values, None)



