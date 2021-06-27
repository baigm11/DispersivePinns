from ImportFile import *

#from Bai.ModelClassTorch2 import fit, StandardFit

pi = math.pi
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def initialize_inputs(len_sys_argv):
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 0

        # Number of training+validation points
        n_coll_ = 2048*4 # 1500 * 9
        n_u_ = 1024 # 1024 * 4
        n_int_ = 0

        # Only for Navier Stokes
        n_object = 0  # useless
        ob = None  # useless

        # Additional Info
        folder_path_ = "Test"
        # additional
        anim_name_ = "Kawa_single"
        point_ = "sobol"  # random
        validation_size_ = 0.0  # useless
        network_properties_ = {
            "hidden_layers": 4,
            "neurons": 20,
            "residual_parameter": 0.1, #0.1
            "kernel_regularizer": 2,
            "regularization_parameter": 0,
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "max_iter": 3,
            "activation": "tanh",
            "optimizer": "LBFGS" #ADAM
        }
        retrain_ = 32
        shuffle_ = False

    elif len_sys_argv == 14:
        print(sys.argv)
        # Random Seed for sampling the dataset
        sampling_seed_ = int(sys.argv[1])

        # Number of training+validation points
        n_coll_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_int_ = int(sys.argv[4])

        # Only for Navier Stokes
        n_object = int(sys.argv[5])
        if sys.argv[6] == "None":
            ob = None
        else:
            ob = sys.argv[6]

        # Additional Info
        folder_path_ = sys.argv[7]
        anim_name_ = sys.argv[8]
        point_ = sys.argv[9]
        validation_size_ = float(sys.argv[10])
        network_properties_ = json.loads(sys.argv[11])
        retrain_ = sys.argv[12]
        if sys.argv[13] == "false":
            shuffle_ = False
        else:
            shuffle_ = True

    else:
        raise ValueError("One input is missing")

    return sampling_seed_, n_coll_, n_u_, n_int_, n_object, ob, folder_path_, anim_name_, point_, validation_size_, network_properties_, retrain_, shuffle_


sampling_seed, N_coll, N_u, N_int, N_object, Ob, folder_path, anim_name, point, validation_size, network_properties, retrain, shuffle = initialize_inputs(len(sys.argv))

if Ec.extrema_values is not None:
    extrema = Ec.extrema_values
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.time_dimensions
    parameter_dimensions = Ec.parameter_dimensions

    print(space_dimensions, time_dimension, parameter_dimensions)
else:
    print("Using free shape. Make sure you have the functions:")
    print("     - add_boundary(n_samples)")
    print("     - add_collocation(n_samples)")
    print("in the Equation file")

    extrema = None
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.time_dimensions
try:
    parameters_values = Ec.parameters_values
    parameter_dimensions = parameters_values.shape[0]
except AttributeError:
    print("No additional parameter found")
    parameters_values = None
    parameter_dimensions = 0

input_dimensions = parameter_dimensions + time_dimension + space_dimensions
output_dimension = Ec.output_dimensions
mode = "none"
if network_properties["epochs"] != 1:
    max_iter = 1
else:
    max_iter = network_properties["max_iter"]


if Ob == "cylinder":
    solid_object = ObjectClass.Cylinder(N_object, 1, input_dimensions, time_dimension, extrema, 1, 0, 0)
elif Ob == "square":
    solid_object = ObjectClass.Square(N_object, 1, input_dimensions, time_dimension, extrema, 2, 2, 0, 0)
else:
    solid_object = None

N_u_train = int(N_u * (1 - validation_size))
N_coll_train = int(N_coll * (1 - validation_size))
N_int_train = int(N_int * (1 - validation_size))
N_object_train = int(N_object * (1 - validation_size))
N_train = N_u_train + N_coll_train + N_int_train + N_object_train

N_u_val = N_u - N_u_train
N_coll_val = N_coll - N_coll_train
N_int_val = N_int - N_int_train
N_object_val = N_object - N_object_train
N_val = N_u_val + N_coll_val + N_int_val + N_object_val

#kdv
if space_dimensions > 0:
    N_b_train = int(N_u_train / (4 * space_dimensions))
else:
    N_b_train = 0
if time_dimension == 1:
    N_i_train = N_u_train - 2 * space_dimensions * N_b_train
elif time_dimension == 0:
    N_b_train = int(N_u_train / (2 * space_dimensions))
    N_i_train = 0
else:
    raise ValueError()

if space_dimensions > 1:
    N_b_val = int(N_u_val / (4 * space_dimensions))
else:
    N_b_val = 0
if time_dimension == 1:
    N_i_val = N_u_val - 2 * space_dimensions * N_b_val
elif time_dimension == 0:
    N_i_val = 0
else:
    raise ValueError()


print("\n######################################")
print("*******Domain Properties********")
print(extrema)

print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_coll_train)
print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
print("Number of internal points: ", N_int_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("*******Info Validation Points********")
print("Number of train collocation points: ", N_coll_val)
print("Number of initial and boundary points: ", N_u_val)
print("Number of internal points: ", N_int_val)
print("Total number of training points: ", N_val)

print("\n######################################")
print("*******Network Properties********")
pprint.pprint(network_properties)
batch_dim = network_properties["batch_size"]

print("\n######################################")
print("*******Dimensions********")
print("Space Dimensions", space_dimensions)
print("Time Dimension", time_dimension)
print("Parameter Dimensions", parameter_dimensions)
print("\n######################################")

if network_properties["optimizer"] == "LBFGS" and network_properties["epochs"] != 1 and network_properties["max_iter"] == 1 and (batch_dim =="full" or batch_dim== N_train):
    print(bcolors.WARNING + "WARNING: you set max_iter=1 and epochs=" + str(network_properties["epochs"]) + " with a LBFGS optimizer.\n"
        "This will work but it is not efficient in full batch mode. Set max_iter = " + str(network_properties["epochs"]) + " and epochs=1. instead" + bcolors.ENDC)

if batch_dim == "full":
    batch_dim = N_train

# ##############################################################################################
# Datasets Creation
training_set_class = DefineDataset(extrema,
                                   parameters_values,
                                   point,
                                   N_coll_train,
                                   N_b_train,
                                   N_i_train,
                                   N_int_train,
                                   batches=batch_dim,
                                   output_dimension=output_dimension,
                                   space_dimensions=space_dimensions,
                                   time_dimensions=time_dimension,
                                   parameter_dimensions=parameter_dimensions,
                                   random_seed=sampling_seed,
                                   obj=solid_object,
                                   shuffle=shuffle)
training_set_class.assemble_dataset()
training_set_no_batches = training_set_class.data_no_batches

'''if validation_size > 0:
    validation_set_class = DefineDataset(extrema, point, N_coll_val, N_b_val, N_i_val, N_int_val, batches=batch_dim,
                                         output_dimension=output_dimension,
                                         space_dimensions=space_dimensions,
                                         time_dimensions=time_dimension,
                                         random_seed=10 * sampling_seed)
    validation_set_class.assemble_dataset()
else:
    validation_set_class = None'''
validation_set_class = None
# ##############################################################################################
# Ignore the commented part
'''
if mode == "IC":
    model_IC = Pinns(input_dimension=input_dimensions + parameter_dimension, output_dimension=output_dimension,
                     network_properties=network_properties, solid_object=solid_object)
    torch.manual_seed(retrain)
    init_xavier(model_IC)
    if torch.cuda.is_available():
        print("Loading model on GPU")
        model_IC.cuda()

    optimizer_IC = optim.LBFGS(model_IC.parameters(), lr=0.8, max_iter=50000, max_eval=50000, history_size=100,
                               line_search_fn="strong_wolfe",
                               tolerance_change=1.0 * np.finfo(float).eps)  # 1.0 * np.finfo(float).eps
    history_IC = fit(model_IC, optimizer_IC, training_set_class, validation_set_clsss=validation_set_class,
                     verbose=True, training_ic=True)

    for param in model_IC.parameters():
        param.requires_grad = False

    additional_models = model_IC

elif mode == "all":
    list_name_models = ["A", "B", "C"]
    model_A = Pinns(input_dimension=input_dimensions + parameter_dimension, output_dimension=output_dimension,
                    network_properties=network_properties, solid_object=solid_object)
    model_B = Pinns(input_dimension=input_dimensions + parameter_dimension, output_dimension=output_dimension,
                    network_properties=network_properties, solid_object=solid_object)
    model_C = Pinns(input_dimension=input_dimensions + parameter_dimension, output_dimension=output_dimension,
                    network_properties=network_properties, solid_object=solid_object)
    torch.manual_seed(retrain)
    init_xavier(model_A)
    init_xavier(model_B)
    init_xavier(model_C)
    if torch.cuda.is_available():
        print("Loading model on GPU")
        model_A.cuda()
        model_B.cuda()
        model_C.cuda()
    list_models = [model_A, model_B, model_C]
    optimizer_A = optim.LBFGS(model_A.parameters(), lr=0.8, max_iter=50000, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_B = optim.LBFGS(model_B.parameters(), lr=0.8, max_iter=50000, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_C = optim.LBFGS(model_C.parameters(), lr=0.8, max_iter=50000, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)

    list_optimizer = [optimizer_A, optimizer_B, optimizer_C]
    trainpartmods(training_set_class, list_optimizer, list_name_models, list_models)

    additional_models = list_models


else:
    additional_models = None'''
# ##############################################################################################
# Model Creation
additional_models = None
model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
              network_properties=network_properties, additional_models=additional_models)
torch.manual_seed(retrain)
init_xavier(model)
if torch.cuda.is_available():
    print("Loading model on GPU")
    model.cuda()

start = time.time()
print("Fitting Model")
model.train()

# ##############################################################################################
# Model Training
optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)  # 1.0 * np.finfo(float).eps
optimizer_ADAM = optim.Adam(model.parameters(), lr=0.00005)
print(network_properties)

if network_properties["optimizer"] == "LBFGS":
    model.optimizer= optimizer_LBFGS
elif network_properties["optimizer"]== "ADAM":
    model.optimizer = optimizer_ADAM
else:
    raise ValueError()

print(network_properties)

if N_coll_train != 0:
    print("bg")
    final_error_train = fit(model, training_set_class, validation_set_clsss=validation_set_class, verbose=True,
                            training_ic=False)
    print("ly")
else:
    final_error_train = StandardFit(model, training_set_class, validation_set_clsss=validation_set_class, verbose=True)
end = time.time() - start

print("\nTraining Time: ", end)

model = model.eval()
final_error_train = float(((10 ** final_error_train) ** 0.5).detach().cpu().numpy())
print("\n################################################")
print("Final Training Loss:", final_error_train)
print("################################################")
final_error_val = None
final_error_test = 0

# ##############################################################################################
# Plotting ang Assessing Performance
images_path = folder_path + "/Images"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)
model_path = folder_path + "/TrainedModel"
os.makedirs(model_path, exist_ok=True)
L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)
Ec.plotting(model, images_path, extrema, solid_object)

end_plotting = time.time() - end

print("\nPlotting and Computing Time: ", end_plotting)

torch.save(model, model_path + "/model.pkl")
with open(model_path + os.sep + "Information.csv", "w") as w:
    keys = list(network_properties.keys())
    vals = list(network_properties.values())
    w.write(keys[0])
    for i in range(1, len(keys)):
        w.write("," + keys[i])
    w.write("\n")
    w.write(str(vals[0]))
    for i in range(1, len(vals)):
        w.write("," + str(vals[i]))

with open(folder_path + '/InfoModel.txt', 'w') as file:
    file.write("Nu_train,"
               "Nf_train,"
               "Nint_train,"
               "validation_size,"
               "train_time,"
               "L2_norm_test,"
               "rel_L2_norm,"
               "error_train,"
               "error_val,"
               "error_test\n")
    file.write(str(N_u_train) + "," +
               str(N_coll_train) + "," +
               str(N_int_train) + "," +
               str(validation_size) + "," +
               str(end) + "," +
               str(L2_test) + "," +
               str(rel_L2_test) + "," +
               str(final_error_train) + "," +
               str(final_error_val) + "," +
               str(final_error_test))



print("\n################################################")
print("saving animation")
print("################################################")

#create animation
#only for kdv(1-dim)
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use('seaborn-pastel')


fig = plt.figure()
ax = plt.axes(xlim=extrema[1, :], ylim=Ec.val_range)
line, = ax.plot([], [], lw=3)

frames = 200

def init():
    line.set_data([], [])
    return line,
def animate(i, model=model):
    model.eval()

    t = extrema[0, 0] + i * (extrema[0, 1] - extrema[0, 0]) / frames

    x = torch.linspace(extrema[1, 0], extrema[1, 1], 1000).reshape(-1, 1)
    t_vec = torch.full(size=(x.shape[0], 1), fill_value=t, dtype=torch.float)
    inputs = torch.cat([t_vec, x], 1)
    out = model(inputs)
    line.set_data(x.detach().numpy(), out.detach().numpy())
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

#anim.save(images_path + '/Kawa_single.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
writergif = PillowWriter(fps=30)
anim.save(images_path + "/" + anim_name + ".gif", writer=writergif)

