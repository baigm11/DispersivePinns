from ImportFile import *

pi = math.pi

# Number of time dimensions
time_dimensions = 1
# Number of space dimensions
space_dimensions = 1
# Number of additional non-space and non-temporal dimensions (for UQ problem for instance)
parameter_dimensions = 0
# Number of output dimensions
output_dimensions = 2
# Domain Extrema
extrema_values = torch.tensor([[-5., 5.],  # Time t; single [-1., 1.]; double [-5., 5.]
                               [-15., 15.]])  # Space x; single [-10., 10.]; double [-15., 15.]
# Additional variable to use here
c = 3

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
    net = network(x_f_train).reshape(-1, 2)
    u = net[:, 0]
    p = net[:, 1]

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0]
    grad_u_t = grad_u[:, 0]
    grad_u_x = grad_u[:, 1]
    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0]), create_graph=True)[0][:, 1]

    grad_p_x = torch.autograd.grad(p, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 1]

    residual1 = grad_u_t + u * grad_u_x + grad_p_x
    residual2 = p - grad_u_xx

    residual = (residual1 ** 2 + residual2 ** 2) ** 0.5
    # grad_u_xxx = torch.autograd.grad(grad_u_xx, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0]), create_graph=True)[0][:, 1]
    # residual = grad_u_t + u * grad_u_x + grad_u_xxx

    return residual


def exact(inputs):
    '''
    Compute the exact solution
    Args:
        inputs: inputs parameter where to compute the exact solution

    Returns: the vector containing the BC at given inputs
    '''

    inputs_cp = inputs.detach().clone()
    t = inputs_cp[:, 0]
    x = inputs_cp[:, 1]
    x.requires_grad = True

    # KdV single soliton
    # u = torch.tensor(3 * c) / torch.cosh(np.sqrt(c) / 2 * (x - c * t)) ** 2

    # KdV double soliton
    a = torch.tensor(.5)
    b = torch.tensor(1.)

    u = 6 * (b - a) \
        * (b / torch.sinh(torch.sqrt(0.5 * b) * (x - 2 * b * t)) ** 2 + a / torch.cosh(torch.sqrt(0.5 * a) * (x - 2 * a * t)) ** 2) \
        / (torch.sqrt(a) * torch.tanh(torch.sqrt(0.5 * a) * (x - 2 * a * t)) - torch.sqrt(b) / torch.tanh(torch.sqrt(0.5 * b) * (x - 2 * b * t))) ** 2

    # computation of p
    grad_u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones(x.shape[0], ), create_graph=True)[0]
    p = torch.autograd.grad(grad_u_x, x, grad_outputs=torch.ones(x.shape[0], ), create_graph=True)[0]

    return torch.cat([u.reshape(-1, 1), p.reshape(-1, 1)], 1).detach()


def ub0(t):
    '''
    Assign Boundary conditions at x=x_left
    Args:
        t: time vector (x is fixed and x=x_left, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns:
    '''
    # Specify tipy of BC: "func" = "Dirichlet
    type_BC = ["func", "func"]
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 0], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    out = exact(inputs)
    return out.reshape(-1, 2), type_BC


def ub1(t):
    '''
    Assign Boundary conditions at x=x_right
    Args:
        t: time vector (x is fixed and x=x_right, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns: the vector containing the BC at given inputs
    '''
    type_BC = ["func", "func"]
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    out = exact(inputs)
    return out.reshape(-1, 2), type_BC


list_of_BC = [[ub0, ub1]]


def u0(x):
    '''
    Assign the initial condition
    Args:
        x: space vector (t is fixed and t=0, so it is not given as input). IC can be function of space only

    Returns: the vector containing the IC at given inputs

    '''

    # KdV single soliton
    # u0 = torch.tensor(3 * c) / torch.cosh(np.sqrt(c) / 2 * (x + c)) ** 2

    # KdV double soliton
    # a = torch.tensor(.5)
    # b = torch.tensor(1.)
    # t0 = torch.full(size=(x.shape[0], 1), fill_value=extrema_values[0, 0], dtype=torch.float)
    #
    # u0 = 6 * (b - a) \
    #     * (b / torch.sinh(torch.sqrt(0.5 * b) * (x - 2 * b * t0)) ** 2 + a / torch.cosh(torch.sqrt(0.5 * a) * (x - 2 * a * t0)) ** 2) \
    #     / (torch.sqrt(a) * torch.tanh(torch.sqrt(0.5 * a) * (x - 2 * a * t0)) - torch.sqrt(b) / torch.tanh(torch.sqrt(0.5 * b) * (x - 2 * b * t0))) ** 2

    t0 = torch.full(size=(x.shape[0], 1), fill_value=extrema_values[0, 0], dtype=torch.float)

    inputs = torch.cat([t0, x], 1)

    u0 = exact(inputs)

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
    L2_test = np.sqrt(np.mean((Exact - test_out)[:, 0] ** 2))
    print("Error Test:", L2_test)

    rel_L2_test = L2_test / np.sqrt(np.mean(Exact[:, 0] ** 2))
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
    scale_vec = np.linspace(0.65, 1.55, len(time_steps) * 2)

    scale_vec_sys = []  # for system
    for i in range(len(time_steps)):
        scale_vec_sys.append(scale_vec[len(time_steps) * i: len(time_steps) * i + 2])


    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    for val, scale in zip(time_steps, scale_vec_sys):
        plot_var = torch.cat([torch.tensor(()).new_full(size=(n, 1), fill_value=val), x], 1)
        plt.plot(x, exact(plot_var)[:, 0], 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=lighten_color('grey', scale[0]), zorder=0)
        plt.plot(x, exact(plot_var)[:, 1], 'b-', linewidth=2, label=r'Exact p, $t=$' + str(val) + r'$s$',color=lighten_color('grey', scale[1]), zorder=0)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 0].detach().numpy(), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=5, color=lighten_color('C0', scale[0]), zorder=10)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 1].detach().numpy(),label=r'Predicted p, $t=$' + str(val) + r'$s$', marker="o", s=5, color=lighten_color('C0', scale[1]), zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend()
    plt.savefig(images_path + "/Samples.png", dpi=500)
