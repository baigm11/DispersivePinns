from ImportFile import *

pi = math.pi

# Number of time dimensions
time_dimensions = 1
# Number of space dimensions
space_dimensions = 1
# Number of additional non-space and non-temporal dimensions (for UQ problem for instance)
parameter_dimensions = 0
# Number of output dimensions
output_dimensions = 2 # Schrodinger!!
# Domain Extrema
extrema_values = torch.tensor([[-0.75, 0.75],  # Time t [-0.75, 0.75]; double [0., 3.]; sech [0., 0.6]
                               [-10., 10.]])  # Space x [-10., 10.]; [-10., 10.]; [-10., 10.]
# Additional variable to use here
c = 3

# val_range = [-1., 9.] # single
# val_range = [-1., 6.] # double

alpha = 1.
q = 2. #single & double soliton
# q = 8. # sech test case
S = 4.

def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error=False):
    '''
    Compute the PDE residual al the given interior collocation points
    Args:
        network: neural network representing solution of the PDE
        x_f_train: collocation interior points
        space_dimensions: not used, to ignore
        solid_object: not used, to ignore
        computing_error: not used, to ignore

    Returns: PDE residual at x_f_train
    '''

    # q = 2.

    x_f_train.requires_grad = True
    # print(x_f_train.shape, network(x_f_train).shape)
    u = network(x_f_train).reshape(-1, 2)
    u_re = u[:, 0]
    u_im = u[:, 1]
    grad_u_t_re = torch.autograd.grad(u_re, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 0]
    grad_u_t_im = torch.autograd.grad(u_im, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 0]
    grad_u_x_re = torch.autograd.grad(u_re, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 1]
    grad_u_x_im = torch.autograd.grad(u_im, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 1]
    grad_u_xx_re = torch.autograd.grad(grad_u_x_re, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 1]
    grad_u_xx_im = torch.autograd.grad(grad_u_x_im, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 1]
    u_sq = u_re ** 2 + u_im ** 2

    residual_re = - grad_u_t_im + grad_u_xx_re + q * u_sq * u_re
    residial_im = grad_u_t_re + grad_u_xx_im + q * u_sq * u_im

    residual = (residual_re ** 2 + residial_im ** 2) ** 0.5

    return residual


def exact(inputs):
    '''
    Compute the exact solution
    Args:
        inputs: inputs parameter where to compute the exact solution

    Returns: the vector containing the BC at given inputs
    '''
    t = inputs[:, 0]
    x = inputs[:, 1]

    # alpha = 1.
    # q = 2.
    # S = 4.

    u = torch.full(size=(t.shape[0], 2), fill_value=0, dtype=torch.float)
    u[:, 0] = alpha * (2 / q) ** 0.5 * torch.cos(0.5 * S * x - 0.25 * (S ** 2 - alpha ** 2) * t) / torch.cosh(alpha * (x - S * t))
    u[:, 1] = alpha * (2 / q) ** 0.5 * torch.sin(0.5 * S * x - 0.25 * (S ** 2 - alpha ** 2) * t) / torch.cosh(alpha * (x - S * t))

    return u.reshape(-1, 2)


def ub0(t):
    '''
    Assign Boundary conditions at x=x_left
    Args:
        t: time vector (x is fixed and x=x_left, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns:
    '''
    # Specify tipy of BC: "func" = "Dirichlet
    type_BC = ["func", "func"]
    # single & sech
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 0], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    out = exact(inputs)

    # double
    # out = torch.full(size=(t.shape[0], 2), fill_value=0, dtype=torch.double)

    return out.reshape(-1, 2), type_BC


def ub1(t):
    '''
    Assign Boundary conditions at x=x_right
    Args:
        t: time vector (x is fixed and x=x_right, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns: the vector containing the BC at given inputs
    '''
    type_BC = ["func", "func"]
    # single & sech
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    out = exact(inputs)

    # double
    # out = torch.full(size=(t.shape[0], 2), fill_value=0, dtype=torch.double)

    return out.reshape(-1, 2), type_BC

list_of_BC = [[ub0, ub1]]

def u0(x):
    '''
    Assign the initial condition
    Args:
        x: space vector (t is fixed and t=0, so it is not given as input). IC can be function of space only

    Returns: the vector containing the IC at given inputs

    '''

    # single soliton
    t0 = torch.full(size=(x.shape[0], 1), fill_value=extrema_values[0, 0], dtype=torch.float)
    inputs = torch.cat([t0, x], 1)
    u0 = exact(inputs)

    # double soliton
    # S1 = torch.tensor(-4.)
    # S2 = torch.tensor(4.)
    # x1 = torch.tensor(6.)
    # x2 = torch.tensor(-6.)
    # x = x.reshape(-1, )
    # u0 = torch.full(size=(x.shape[0], 2), fill_value=0, dtype=torch.float)
    # u0[:, 0] = alpha * (2 / q) ** 0.5 * torch.cos(0.5 * S1 * (x - x1)) / torch.cosh(alpha * (x - x1)) \
    #         + alpha * (2 / q) ** 0.5 * torch.cos(0.5 * S2 * (x - x2)) / torch.cosh(alpha * (x - x2))
    # u0[:, 1] = alpha * (2 / q) ** 0.5 * torch.sin(0.5 * S1 * (x - x1)) / torch.cosh(alpha * (x - x1)) \
    #         + alpha * (2 / q) ** 0.5 * torch.sin(0.5 * S2 * (x - x2)) / torch.cosh(alpha * (x - x2))

    #sech test
    # x = x.reshape(-1, )
    # u0 = torch.full(size=(x.shape[0], 2), fill_value=0, dtype=torch.float)
    # u0[:, 0] = 1 / torch.cosh(x) # u0[:, 1] = 0
    # u0[:, 1] = 0.

    return u0.reshape(-1, 2)


def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    '''
    Compute the generalization error
    Args:
        model: neural network approximating the solution
        extrema: extrema of the domain
        images_path: path where to save some pictures

    Returns: absolute and relative L2 norm of the error solution
    '''
    model.eval()
    test_inp = convert(torch.rand([100000, extrema.shape[0]]), extrema)
    Exact = (exact(test_inp)).numpy()
    test_out = model(test_inp).detach().numpy()
    assert (Exact.shape[1] == test_out.shape[1])
    L2_test = np.sqrt(np.mean(((Exact - test_out) ** 2).sum(1)))
    L2_test_env = np.sqrt(np.abs(np.mean(((Exact ** 2).sum(1)) - (test_out ** 2).sum(1)))) # envelope
    print("Error Test:", L2_test, L2_test_env)

    rel_L2_test = L2_test / np.sqrt(np.mean((Exact ** 2).sum(1)))
    rel_L2_test_env = L2_test_env / np.sqrt(np.mean((Exact ** 2).sum(1))) # envelope
    print("Relative Error Test:", rel_L2_test, rel_L2_test_env)

    if images_path is not None:
        plt.figure()
        plt.grid(True, which="both", ls=":")
        plt.scatter(Exact, test_out)
        plt.xlabel(r'Exact Values')
        plt.ylabel(r'Predicted Values')
        plt.savefig(images_path + "/Score.png", dpi=400)
    return L2_test, rel_L2_test


def plotting(model, images_path, extrema, solid):
    '''
    Function to make plots
    Args:
        model: neural network approximating the solution
        images_path: path where to save plots
        extrema:  extrema of the domain
        solid:
    '''
    model.cpu()
    model = model.eval()
    n = 500
    x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], n), [n, 1])
    time_steps = extrema_values[0, :].detach().numpy()    #[-5., 5.]
    scale_vec = np.linspace(0.35, 2., len(time_steps) * 3) # 0.65, 1.55; 0.35, 2.
    scale_vec_c = [] # for complex function
    for i in range(len(time_steps)):
        scale_vec_c.append(scale_vec[len(time_steps) * i : len(time_steps) * i + 3])

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    for val, scale in zip(time_steps, scale_vec_c):
        plot_var = torch.cat([torch.tensor(()).new_full(size=(n, 1), fill_value=val), x], 1)

        ext = (exact(plot_var)[:, 0]**2 + exact(plot_var)[:, 1]**2)**0.5 # Schrodinger
        plt_var = (model(plot_var)[:, 0] ** 2 + model(plot_var)[:, 1] ** 2) ** 0.5  # Schrodinger

        # only for single soliton(have exact sol)
        plt.plot(x, ext.reshape(-1, 1), 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$',color=lighten_color('grey', scale[0]), zorder=0)
        plt.plot(x, exact(plot_var)[:, 0].reshape(-1, 1), 'b-', linewidth=2, label=r'Exact Re, $t=$' + str(val) + r'$s$', color=lighten_color('grey', scale[1]), zorder=0)
        plt.plot(x, exact(plot_var)[:, 1].reshape(-1, 1), 'b-', linewidth=2, label=r'Exact Im, $t=$' + str(val) + r'$s$',color=lighten_color('grey', scale[2]), zorder=0)

        plt.scatter(plot_var[:, 1].detach().numpy(), plt_var.detach().numpy().reshape(-1, 1), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=5,
                    color=lighten_color('C0', scale[0]), zorder=10)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 0].detach().numpy(), label=r'Predicted Re, $t=$' + str(val) + r'$s$', marker="o", s=5,
                    color=lighten_color('C0', scale[1]), zorder=10)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 1].detach().numpy(), label=r'Predicted Im, $t=$' + str(val) + r'$s$', marker="o", s=5,
                    color=lighten_color('C0', scale[2]), zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend(loc=2) # loc=2 for single soliton
    plt.savefig(images_path + "/Samples.png", dpi=500)
