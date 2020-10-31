import os
import tensorflow as tf
import numpy as np
import EquationClassBurg as Ec
import math

def seed_random_number(seed):
    # see https://stackoverflow.com/a/52897216
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_model(best_model, information, folderpath):

    best_model.save(folderpath + os.sep + "model.h5")
    # Save info
    with open(folderpath + os.sep + "Information.csv", "w") as w:
        keys = list(information.keys())
        vals = list(information.values())
        w.write(keys[0])
        for i in range(1, len(keys)):
            w.write(","+keys[i])
        w.write("\n")
        w.write(str(vals[0]))
        for i in range(1, len(vals)):
            w.write("," + str(vals[i]))


def load_model(folder_name):
    folder_path = folder_name + os.sep + "model_best.h5"
    print(folder_path)
    loaded_model = tf.keras.models.load_model(folder_path, compile=False)
    return loaded_model


def loss_function(x_u_train, u_train, x_f_train, network, lambda_res=1.0):
    u_pred = network(x_u_train)
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x_f_train)
        with tf.GradientTape(persistent=True) as t:
            t.watch(x_f_train)
            out = network(x_f_train)
            out_sq = network(x_f_train)*network(x_f_train)/2.0
            gradient = t.gradient(out, x_f_train)
            gradient_squ = t.gradient(out_sq, x_f_train)
        grad2_x = t2.gradient(gradient[:, 1], x_f_train)
        grad2_t = t2.gradient(gradient[:, 0], x_f_train)
    #print(x_f_train)

    # Equation form
    #pred = Ec.equation(u=out, dudt=tf.cast(gradient[:, 1], dtype=tf.float32), dudx=tf.cast(gradient[:, 0],dtype=tf.float32), dudx2=tf.cast(grad2_x[:, 0],dtype=tf.float32), s=None)
    pred = gradient[:, 0] + (gradient_squ[:,1]) - (0.01/math.pi)*grad2_x[:,1]
    #pred = tf.cast(gradient[:, 1], dtype=tf.float32) + out*gradient[:, 0] - - (0.01/math.pi)*grad2_x[:,0]
    #print(tf.reduce_mean(tf.square(pred)), tf.reduce_mean(tf.square(u_train - u_pred)))
    loss_value = tf.add(tf.reduce_mean(tf.square(u_train - u_pred)), tf.reduce_mean(tf.square(pred)))

    return tf.cast(loss_value, dtype=tf.float32)


'''

def function_factory(model, loss_f, x_u_train, u_train, x_f_train, lambda_res):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    #@tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):

            model.trainable_variables[i].assign(tf.cast(tf.reshape(param, shape), dtype=tf.float32))

    # now create a function that will be returned by this factory
    #@tf.function
    def f(params_1d):
        """
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss_f(x_u_train, u_train, x_f_train, model, lambda_res)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    return f

'''


def flatten_trainable_variables(trainable_vars):
    shapes = tf.shape_n(trainable_vars)

    count = 0
    idx = []# stitch indices
    part = []# partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    flatten_variables = tf.dynamic_stitch(idx, trainable_vars)

    return flatten_variables, shapes, part


def reshape_trainable_variables(flatten_variables, shapes, part):
    n_tensors = len(shapes)

    variables = tf.dynamic_partition(flatten_variables, part, n_tensors)

    trainable_variables = list()
    for shape, param in zip(shapes, variables):
        trainable_variables.append(tf.reshape(param, shape))

    return trainable_variables


def f(inputs, shapes, part, model, x_u_train, u_train, x_f_train, loss, folder_path):

    reshaped_input = reshape_trainable_variables(inputs, shapes, part)

    with tf.GradientTape() as tape:
        for k in range(len(reshaped_input)):
            model.trainable_variables[k].assign(tf.cast(reshaped_input[k], dtype=tf.float32))
        loss_val = loss(x_u_train, u_train, x_f_train, model)
    grads = tape.gradient(loss_val, model.trainable_variables)
    flatten_grad = flatten_trainable_variables(grads)[0].numpy()
    print(loss_val)
    #model.save(folder_path + "/TrainedModel" + os.sep + "model_best.h5")
    return float(loss_val), flatten_grad.astype('float64')


def grad(y, x, network, n_collocation, lambda_res=1.0):
    with tf.GradientTape() as tape:
        loss_value = loss_function(y, x, network, n_collocation, lambda_res=lambda_res)
    return loss_value, tape.gradient(loss_value, network.trainable_variables)



def custom_fit(network, train_set, val_set, optimizer, num_epochs, n_collocation, n_boundary, n_initial, n_internal, verbose=False, monitor="val_loss", lambda_res=1.0):
    train_loss_results = []
    val_loss_results = []

    best_value = np.inf

    n_tot = n_collocation + 2 * n_boundary + n_initial + n_internal

    for epoch in range(num_epochs):

        train_loss_batches = []

        for x, y in train_set:
            # print(tf.math.equal(list(train_set)[0][0],x))
            if verbose:
                tf.print("------------------------------------------------------")
            batch_dim = tf.shape(x)[0]
            # fraction_bound = int(batch_dim * n_boundary / n_tot)
            # fraction_internal = int(batch_dim * n_internal / n_tot)
            fraction_coll = int(batch_dim * n_collocation / n_tot)
            # fraction_initial = int(batch_dim - 2 * fraction_bound - fraction_coll - fraction_internal)

            loss_v, grads = grad(y, x, network, fraction_coll, lambda_res=lambda_res)

            if verbose:
                tf.print("Batches Losses: ", loss_v.numpy())

            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            train_loss_batches.append(loss_v)

        mean_over_batches = tf.math.reduce_mean(train_loss_batches)
        train_loss_results.append(mean_over_batches)
        if validation_size != 0.0:
            loss_v_val = loss_function(list(val_set)[0][1], list(val_set)[0][0], network, n_collocation, lambda_res=lambda_res)

            val_loss_results.append(loss_v_val)

        if monitor == "val_loss" and validation_size != 0.0:
            candidate_value = loss_v_val
        elif monitor == "train_loss":
            candidate_value = mean_over_batches.numpy()
        else:
            raise ValueError()

        if candidate_value <= best_value:
            best_value = candidate_value
            network.save(folder_path + "/TrainedModel" + os.sep + "model_best.h5")

            print("Saving")

        if verbose:
            tf.print("Epoch: ", epoch,
                     "Train Losses: ",
                     tf.math.reduce_mean(train_loss_batches).numpy(),
                     )
            if validation_size != 0.0:
                tf.print("Epoch: ", epoch,
                         "Validation Losses: ",
                         loss_v_val.numpy(),
                         )

    tf.print(best_value)
    # network.save(folder_path + "/TrainedModel" + os.sep + "model_best.h5")
    return train_loss_results, val_loss_results


