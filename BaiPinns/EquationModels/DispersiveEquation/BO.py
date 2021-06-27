from ImportFile import *

import EquationModels.DispersiveEquation.peakon_lim as pl
import EquationModels.DispersiveEquation.peakon_lim_double as pld

pi = math.pi

# Number of time dimensions
time_dimensions = 1
# Number of space dimensions
space_dimensions = 1
# Number of additional non-space and non-temporal dimensions (for UQ problem for instance)
parameter_dimensions = 0
# Number of output dimensions
output_dimensions = 1
# Domain Extrema
extrema_values = torch.tensor([[0., 50.],  # Time t; single peri [0., 50.]; double line [-1.5, 1.5]; double peri
                               [-15., 15.]])  # Space xl single peri [-15., 15.]; double line [-10., 10.]; double peri
# Additional variable to use here
c = 0.25 #delta x = c * delta t = 12.5

# val_range = [0.1, 1.] #BO single peri soliton
# val_range = [-0.1, 8.] #BO dpuble line soliton
# val_range = [-0.1, 7.] #BO double peri soliton

#half period for single soliton
L = 15. #single peri
# L = 10. #double peri

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
    u_sq = 0.5 * u * u
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0]
    grad_u_sq_x = torch.autograd.grad(u_sq, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ), create_graph=True)[0][:, 1]
    grad_u_t = grad_u[:, 0]
    grad_u_x = grad_u[:, 1]
    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0]), create_graph=True)[0][:, 1]


    ratio = (extrema_values[0, 1] - extrema_values[0, 0]) / (extrema_values[1, 1] - extrema_values[1, 0])

    #get grid info in two directions
    dir1 = len(np.unique(x_f_train[:, 0].detach().numpy()))
    dir2 = len(np.unique(x_f_train[:, 1].detach().numpy()))
    assert x_f_train.shape == torch.Size([dir1 * dir2, 2])
    assert dir2 % 2 == 1

    #middle index
    idx_m = (int)((dir2 - 1) / 2)

    int_shape = [dir1, dir2] #dt, dx
    # print(int_shape)
    # print(x_f_train.shape)
    # print(x_f_train.reshape(dir1, dir2, 2))

    # double, for extended y, don't need to comment for single peri
    amp = 4  # amplfication number, i.e. from +-25 to +-100
    int_shape_ext = [dir1, (dir2 - 1) * amp + 1]

    #single & double
    y = torch.linspace(extrema_values[1, 0], extrema_values[1, 1], int_shape[1])

    #double & single, extended y, don't need to comment for single peri
    y_ext = torch.linspace(extrema_values[1, 0] * amp, extrema_values[1, 1] * amp, int_shape_ext[1])

    # not used, extend data for computing Hilbert trans
    # t_ext = torch.linspace(extrema_values[0, 0], extrema_values[0, 1], int_shape_ext[0])
    # x_f_train_ext = torch.from_numpy(np.array([[t_i, y_i] for t_i in t_ext for y_i in y_ext])).type(torch.FloatTensor).reshape(int_shape_ext[0] * int_shape_ext[1], 2)
    # x_f_train_ext.requires_grad = True
    # u_ext = network(x_f_train_ext).reshape(-1, )
    # grad_u_ext = torch.autograd.grad(u_ext, x_f_train_ext, grad_outputs=torch.ones(x_f_train_ext.shape[0], ), create_graph=True)[0]
    # grad_u_x_ext = grad_u_ext[:, 1]
    # grad_u_xx_ext = torch.autograd.grad(grad_u_x_ext, x_f_train_ext, grad_outputs=torch.ones(x_f_train_ext.shape[0]), create_graph=True)[0][:, 1]
    # # print(y)

    #test the consistency of x_f_train and convolution grid
    # t = torch.linspace(extrema_values[0, 0], extrema_values[0, 1], int_shape[0])
    # grid = torch.from_numpy(np.array([[t_i, y_i] for t_i in t for y_i in y]).reshape(t.shape[0] * y.shape[0],2)).type(torch.FloatTensor)
    # print((grid - x_f_train).abs().sum())

    #single
    conv_coef = 1 / torch.tan(pi * y / (2 * L))
    conv_coef[idx_m] = 0

    # double
    # conv_coef = 1 / y_ext
    # conv_coef[idx_m * amp] = 0
    # # print(conv_coef)

    grad_u_xx.reshape(int_shape)

    H_trans = torch.zeros(int_shape, dtype=torch.float)

    for i in range(int_shape[0]):
        #single, use [:-1] to count boudnary only once
        zeros = torch.full(size=(int_shape[1] - 1, 1), fill_value=0., dtype=torch.double)
        v = grad_u_xx.reshape(int_shape)[i, :].reshape(int_shape[1], 1)[:-1]
        w = conv_coef.reshape(int_shape[1], 1)[:-1].roll(-idx_m) #regularization

        #double
        # zeros = torch.full(size=(int_shape_ext[1], 1), fill_value=0., dtype=torch.double)
        # zeros_append = torch.full(size=(idx_m * (amp - 1), 1), fill_value=0., dtype=torch.double)
        # # v = grad_u_xx_ext.reshape(int_shape_ext)[i, :].reshape(int_shape_ext[1], 1) # exact ext
        # v = grad_u_xx.reshape(int_shape)[i, :].reshape(int_shape[1], 1)
        # v = torch.cat([zeros_append, v, zeros_append], 0) # zero padding
        # w = conv_coef.reshape(int_shape_ext[1], 1).roll(-idx_m * amp)  # regularization, zero ext


        # test the correctness of fft
        # zeros = torch.full(size=(5, 1), fill_value=0., dtype=torch.double)
        # v = torch.tensor([1., 2., 3., 1, 2]).reshape(5, 1)
        # w = torch.tensor([1., 1., 2., 2, 2]).reshape(5, 1)

        assert not torch.isnan(w).any()

        v1 = torch.cat((v, zeros), 1)
        w1 = torch.cat((w, zeros), 1)

        v2 = torch.fft(v1, 1, normalized=True)
        w2 = torch.fft(w1, 1, normalized=True)

        v3 = torch.view_as_complex(v2)
        w3 = torch.view_as_complex(w2)

        c = torch.view_as_real(v3 * w3)

        out = torch.ifft(c, 1, normalized=True)
        assert not torch.isnan(out).any()
        # assert the imaginary part is 0.
        assert(out[:, 1].sum() <= 10e-05)

        #double
        # # print(out)
        # out = out[idx_m * (amp - 1):idx_m * (amp - 1) + int_shape[1], :]
        # assert out.shape[0] == int_shape[1]

        #single
        H_trans[i, :-1] = out[:, 0] * int_shape[1] ** 0.5
        H_trans[i, -1] = H_trans[i, 1]

        #double
        # H_trans[i, :] = out[:, 0] * int_shape_ext[1] ** 0.5

    #single
    H_trans = H_trans / int_shape[1]

    # double, different weight
    # H_trans = (extrema_values[1, 1] - extrema_values[1, 0]) / (pi * int_shape[1]) * H_trans

    # print(H_trans)

    residual = grad_u_t.reshape(-1, ) + u.reshape(-1, ) * grad_u_x.reshape(-1, ) - H_trans.reshape(-1, )

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


    # BO single peri soliton
    x0 = 0.
    delta = pi / (c * L)
    u = (2 * c * delta * delta) / (1 - np.sqrt(1 - delta * delta) * torch.cos(c * delta * (x - c * t - x0)))

    # BO double line soliton
    # c1 = 2.
    # c2 = 1.
    # lambda1 = x - c1 * t
    # lambda2 = x - c2 * t
    # u = 4*c1*c2*(c1*lambda1**2+c2*lambda2**2+(c1+c2)**3/(c1*c2*(c1-c2)**2)) \
    #     / ((c1*c2*lambda1*lambda2-(c1+c2)**2/(c1-c2)**2)**2+(c1*lambda1+c2*lambda2)**2)


    # BO double peri soliton
    # L = 10. #half period
    # k1 = pi / L
    # k2 = 2 * pi / L
    #
    # phi1 = torch.tensor(0.15)  # 4
    # phi2 = torch.tensor(1.)  # 1.3
    #
    # c1 = k1 / torch.tan(phi1)
    # c2 = k2 / torch.tan(phi2)
    #
    # eA12 = ((c1 - c2) ** 2 - (k1 - k2) ** 2) / ((c1 - c2) ** 2 - (k1 + k2) ** 2)
    #
    # xi1 = k1 * (x - c1 * t)
    # xi2 = k2 * (x - c2 * t)
    #
    # u1 = eA12**0.5*(k1+k2)*torch.sinh(phi1+phi2) \
    #      + eA12**(-0.5)*(k1-k2)*torch.sinh(phi1-phi2) \
    #      + 2*(k1*torch.sinh(phi1)*torch.cos(xi2) + k2*torch.sinh(phi2)*torch.cos(xi1))
    #
    # u2 = eA12**0.5*(torch.cosh(phi1+phi2)+torch.cos(xi1+xi2)) \
    #      + eA12**(-0.5)*(torch.cosh(phi1-phi2)+torch.cos(xi1-xi2)) \
    #      + 2 * (np.cosh(phi2)*torch.cos(xi1) + torch.cosh(phi1)*torch.cos(xi2))
    #
    # u = 2 * u1 / u2

    return u.reshape(-1, 1)


def ub0(t):
    '''
    Assign Boundary conditions at x=x_left
    Args:
        t: time vector (x is fixed and x=x_left, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns:
    '''
    # Specify tipy of BC: "func" = "Dirichlet periodic
    type_BC = ["func"] # periodic for single peri; func for double line
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 0], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    out = exact(inputs)
    return out.reshape(-1, 1), type_BC


def ub1(t):
    '''
    Assign Boundary conditions at x=x_right
    Args:
        t: time vector (x is fixed and x=x_right, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns: the vector containing the BC at given inputs
    '''
    type_BC = ["func"] # periodic for single peri; func for double line
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    out = exact(inputs)
    return out.reshape(-1, 1), type_BC


def ub2(t):
    '''
    Assign Boundary conditions at x=x_right
    Args:
        t: time vector (x is fixed and x=x_right, so it is not given as input). BC can be function of time only for 1D space dimensions

    Returns: the vector containing the BC at given inputs
    '''
    type_BC = ["der"]
    x0 = torch.full(size=(t.shape[0], 1), fill_value=extrema_values[1, 1], dtype=torch.double)
    inputs = torch.cat([t, x0], 1)
    inputs.requires_grad_(True)

    u = exact(inputs).reshape(-1, )
    grad_u = torch.autograd.grad(u, inputs, grad_outputs=torch.ones(inputs.shape[0], ), create_graph=True)[0]

    grad_u_x = grad_u[:, 1]

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

    # BO single peri soliton
    # x0 = 0.
    # delta = pi / (c * L)
    # u0 = (2 * c * delta * delta) / (1 - np.sqrt(1 - delta * delta) * torch.cos(c * delta * (x - c * extrema_values[0, 0] - x0)))

    # BO double line soliton
    c1 = 2.
    c2 = 1.
    t = extrema_values[0, 0]
    lambda1 = x - c1 * t
    lambda2 = x - c2 * t
    u0 = 4 * c1 * c2 * (c1 * lambda1 ** 2 + c2 * lambda2 ** 2 + (c1 + c2) ** 3 / (c1 * c2 * (c1 - c2) ** 2)) \
        / ((c1 * c2 * lambda1 * lambda2 - (c1 + c2) ** 2 / (c1 - c2) ** 2) ** 2 + (c1 * lambda1 + c2 * lambda2) ** 2)


    # t = extrema_values[0, 0]
    # # BO double peri soliton
    # L = 10.  # half period
    # k1 = pi / L
    # k2 = 2 * pi / L
    #
    # phi1 = torch.tensor(0.15)  # 4
    # phi2 = torch.tensor(1.)  # 1.3
    #
    # c1 = k1 / torch.tan(phi1)
    # c2 = k2 / torch.tan(phi2)
    #
    # eA12 = ((c1 - c2) ** 2 - (k1 - k2) ** 2) / ((c1 - c2) ** 2 - (k1 + k2) ** 2)
    #
    # xi1 = k1 * (x - c1 * t)
    # xi2 = k2 * (x - c2 * t)
    #
    # u1 = eA12 ** 0.5 * (k1 + k2) * torch.sinh(phi1 + phi2) \
    #      + eA12 ** (-0.5) * (k1 - k2) * torch.sinh(phi1 - phi2) \
    #      + 2 * (k1 * torch.sinh(phi1) * torch.cos(xi2) + k2 * torch.sinh(phi2) * torch.cos(xi1))
    #
    # u2 = eA12 ** 0.5 * (torch.cosh(phi1 + phi2) + torch.cos(xi1 + xi2)) \
    #      + eA12 ** (-0.5) * (torch.cosh(phi1 - phi2) + torch.cos(xi1 - xi2)) \
    #      + 2 * (np.cosh(phi2) * torch.cos(xi1) + torch.cosh(phi1) * torch.cos(xi2))
    #
    # u0 = 2 * u1 / u2

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
    model.eval()
    test_inp = convert(torch.rand([10000, extrema.shape[0]]), extrema) #100000
    Exact = (exact(test_inp)).numpy()
    test_out = model(test_inp).detach().numpy()
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
        plt.plot(x, exact(plot_var), 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=lighten_color('grey', scale), zorder=0)
        plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var).detach().numpy(), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                    color=lighten_color('C0', scale), zorder=10)

    plt.xlabel(r'$x$')
    plt.ylabel(r'u')
    plt.legend()
    plt.savefig(images_path + "/BO_double_peri_Samples.png", dpi=500)
