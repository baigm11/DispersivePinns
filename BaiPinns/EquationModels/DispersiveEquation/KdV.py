from ImportFile import *

pi = math.pi

# Number of time dimensions
time_dimensions = 1
# Number of space dimensions
space_dimensions = 1
# Number of additional non-space and non-temporal dimensions (for UQ problem for instance)
parameter_dimensions = 3
# Number of output dimensions
output_dimensions = 1
# Domain Extrema
extrema_values = torch.tensor([[-1., 1.],  # Time t; single [-1., 1.]; double [-5., 5.]
                               [-10., 10.]])  # Space x; single [-10., 10.]; double [-15., 15.]
# Additional variable to use here
# c = 3

# UQ single
parameters_values = torch.tensor([[8.7, 9.3], # alpha
                  [-0.4, 0.4], # beta
                    [0.9, 1.1], # gamma
                  [0.9, 1.1]]) # kappa
parameter_dimensions = 4

# UQ double
# parameters_values = torch.tensor([[0.4, 0.6], # a
#                   [0.9, 1.1]]) # b
# parameter_dimensions = 2

# val_range = [-1., 9.] # single
# val_range = [-1., 6.] # double

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

    x_f_train.requires_grad = True
    u = network(x_f_train).reshape(-1, )
    sp = x_f_train.shape
    u_sq = 0.5 * u * u
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(x_f_train[:, 0]), create_graph=True)[0]
    grad_u_sq_x = torch.autograd.grad(u_sq, x_f_train, grad_outputs=torch.ones_like(x_f_train[:, 0]), create_graph=True)[0][:, 1]
    grad_u_t = grad_u[:, 0]
    grad_u_x = grad_u[:, 1]
    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(x_f_train[:, 0]), create_graph=True)[0][:, 1]
    grad_u_xxx = torch.autograd.grad(grad_u_xx, x_f_train, grad_outputs=torch.ones_like(x_f_train[:, 0]), create_graph=True)[0][:, 1]

    # UQ single
    kappa = x_f_train[:, 5].reshape(-1, )
    gamma = x_f_train[:, 4].reshape(-1, )
    residual = grad_u_t.reshape(-1, ) + gamma * grad_u_sq_x.reshape(-1, ) + kappa * grad_u_xxx.reshape(-1, )

    # residual = grad_u_t.reshape(-1, ) + grad_u_sq_x.reshape(-1, ) + grad_u_xxx.reshape(-1, )

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

    # KdV single soliton
    # u = torch.tensor(3 * c) / torch.cosh(np.sqrt(c) / 2 * (x - c * t)) ** 2
    #UQ single
    alpha = inputs[:, 2]
    beta = inputs[:, 3]
    # beta = 0
    gamma = inputs[:, 4]
    kappa = inputs[:, 5]
    u = beta + (alpha - beta) / torch.cosh(np.sqrt((alpha - beta) / (12 * kappa)) * (x - (beta + (alpha - beta) / 3) * t)) ** 2
    u = u / gamma



    # KdV double soliton
    # a = torch.tensor(.5)
    # b = torch.tensor(1.)

    # UQ double
    # a = inputs[:, 2]
    # b = inputs[:, 3]
    #
    # u = 6 * (b - a) \
    #     * (b / torch.sinh(torch.sqrt(0.5 * b) * (x - 2 * b * t)) ** 2 + a / torch.cosh(torch.sqrt(0.5 * a) * (x - 2 * a * t)) ** 2) \
    #     / (torch.sqrt(a) * torch.tanh(torch.sqrt(0.5 * a) * (x - 2 * a * t)) - torch.sqrt(b) / torch.tanh(torch.sqrt(0.5 * b) * (x - 2 * b * t))) ** 2


    return u.reshape(-1, 1)


def ub0(t):
    '''
    Assign Boundary conditions at x=x_left
    Args:
        t: time vector (x is fixed and x=x_left, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns:
    '''
    # Specify tipy of BC: "func" = "Dirichlet
    type_BC = ["func"]
    # x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 0], dtype=torch.double)

    # UQ single
    t0 = t[:, 0].reshape(-1, 1)  # !!
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 0], dtype=torch.double)
    x1 = t[:, 1].reshape(-1, 1)
    x2 = t[:, 2].reshape(-1, 1)
    x3 = t[:, 3].reshape(-1, 1)
    x4 = t[:, 4].reshape(-1, 1)
    inputs = torch.cat([t0, x0, x1, x2, x3, x4], 1)

    # UQ double
    # t0 = t[:, 0].reshape(-1, 1)  # !!
    # x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 0], dtype=torch.double)
    # x1 = t[:, 1].reshape(-1, 1)
    # x2 = t[:, 2].reshape(-1, 1)
    # inputs = torch.cat([t0, x0, x1, x2], 1)

    out = exact(inputs)

    # out = torch.full(size=(t.shape[0], 1), fill_value=0, dtype=torch.double)
    return out.reshape(-1, 1), type_BC


def ub1(t):
    '''
    Assign Boundary conditions at x=x_right
    Args:
        t: time vector (x is fixed and x=x_right, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns: the vector containing the BC at given inputs
    '''
    type_BC = ["func"]
    # x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)

    # UQ single
    t0 = t[:, 0].reshape(-1, 1)  # !!
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    x1 = t[:, 1].reshape(-1, 1)
    x2 = t[:, 2].reshape(-1, 1)
    x3 = t[:, 3].reshape(-1, 1)
    x4 = t[:, 4].reshape(-1, 1)
    inputs = torch.cat([t0, x0, x1, x2, x3, x4], 1)

    # UQ double
    # t0 = t[:, 0].reshape(-1, 1)  # !!
    # x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    # x1 = t[:, 1].reshape(-1, 1)
    # x2 = t[:, 2].reshape(-1, 1)
    # inputs = torch.cat([t0, x0, x1, x2], 1)

    out = exact(inputs)


    # out = torch.full(size=(t.shape[0], 1), fill_value=0, dtype=torch.double)
    return out.reshape(-1, 1), type_BC


def ub2(t):
    '''
    Assign Boundary conditions at x=x_right
    Args:
        t: time vector (x is fixed and x=x_right, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns: the vector containing the BC at given inputs
    '''
    type_BC = ["der"]
    # x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    # inputs = torch.cat([t, x0], 1)
    # UQ single
    t0 = t[:, 0].reshape(-1, 1)  # !!
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    x1 = t[:, 1].reshape(-1, 1)
    x2 = t[:, 2].reshape(-1, 1)
    x3 = t[:, 3].reshape(-1, 1)
    x4 = t[:, 4].reshape(-1, 1)
    inputs = torch.cat([t0, x0, x1, x2, x3, x4], 1)

    # UQ double
    # t0 = t[:, 0].reshape(-1, 1)  # !!
    # x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    # x1 = t[:, 1].reshape(-1, 1)
    # x2 = t[:, 2].reshape(-1, 1)
    # inputs = torch.cat([t0, x0, x1, x2], 1)

    inputs.requires_grad_(True)

    u = exact(inputs).reshape(-1, )
    grad_u = torch.autograd.grad(u, inputs, grad_outputs=torch.ones(inputs.shape[0], ), create_graph=True)[0]

    grad_u_x = grad_u[:, 1]

    # grad_u_x = torch.full(size=(t.shape[0], 1), fill_value=0, dtype=torch.double)

    return grad_u_x.reshape(-1, 1), type_BC


# List containing the BC at x=x_left and x=x_rigth. For many space dimension then:
# list_of_BC = [[ub0x, ub1x],
#               [ub0y, ub1y]],

list_of_BC = [[ub0, ub1, ub2]]


def u0(x):
    '''
    Assign the initial condition
    Args:
        x: space vector (t is fixed and t=0, so it is not given as input). IC can be function of space only

    Returns: the vector containing the IC at given inputs

    '''

    # KdV single soliton
    # u0 = torch.tensor(3 * c) / torch.cosh(np.sqrt(c) / 2 * (x + c)) ** 2
    # UQ single
    # t0 = torch.full(size=(x.shape[0], 1), fill_value=extrema_values[0, 0], dtype=torch.float)
    # inputs = torch.cat([t0, x], 1)
    # u0 = exact(inputs)


    # KdV double soliton
    # a = torch.tensor(.5)
    # b = torch.tensor(1.)
    # t0 = torch.full(size=(x.shape[0], 1), fill_value=extrema_values[0, 0], dtype=torch.float)
    #
    # u0 = 6 * (b - a) \
    #     * (b / torch.sinh(torch.sqrt(0.5 * b) * (x - 2 * b * t0)) ** 2 + a / torch.cosh(torch.sqrt(0.5 * a) * (x - 2 * a * t0)) ** 2) \
    #     / (torch.sqrt(a) * torch.tanh(torch.sqrt(0.5 * a) * (x - 2 * a * t0)) - torch.sqrt(b) / torch.tanh(torch.sqrt(0.5 * b) * (x - 2 * b * t0))) ** 2

    # UQ single & double
    t0 = torch.full(size=(x.shape[0], 1), fill_value=extrema_values[0, 0], dtype=torch.float)
    inputs = torch.cat([t0, x], 1)
    u0 = exact(inputs)

    return u0.reshape(-1, 1)


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
    model = model.to("cpu") # otherwise error on Leonhard
    model.eval()
    test_inp = convert(torch.rand([100000, extrema.shape[0]]), extrema)
    # UQ
    test_inp1 = convert(torch.rand([100000, parameters_values.shape[0]]), parameters_values)
    test_inp = torch.cat([test_inp, test_inp1], 1)
    Exact = (exact(test_inp)).numpy()
    test_out = (model(test_inp)).detach().numpy() # .cpu()
    assert (Exact.shape[1] == test_out.shape[1])
    L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
    print("Error Test:", L2_test)

    rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
    print("Relative Error Test:", rel_L2_test)

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
    scale_vec = np.linspace(0.65, 1.55, len(time_steps))

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    for val, scale in zip(time_steps, scale_vec):
        plot_var = torch.cat([torch.tensor(()).new_full(size=(n, 1), fill_value=val), x], 1)

        # UQ single
        alpha = torch.full(size=(x.shape[0], 1), fill_value=9., dtype=torch.float)
        beta = torch.full(size=(x.shape[0], 1), fill_value=-0., dtype=torch.float)
        gamma = torch.full(size=(x.shape[0], 1), fill_value=1., dtype=torch.float)
        kappa = torch.full(size=(x.shape[0], 1), fill_value=1., dtype=torch.float)
        plot_var = torch.cat([plot_var, alpha, beta, gamma, kappa], 1)

        # UQ double
        # a = torch.full(size=(x.shape[0], 1), fill_value=0.5, dtype=torch.float)
        # b = torch.full(size=(x.shape[0], 1), fill_value=1.0, dtype=torch.float)
        # plot_var = torch.cat([plot_var, a, b], 1)

        plt.plot(x, exact(plot_var), 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=lighten_color('grey', scale), zorder=0)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var).detach().numpy(), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                    color=lighten_color('C0', scale), zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend()
    plt.savefig(images_path + "/Samples.png", dpi=500)
