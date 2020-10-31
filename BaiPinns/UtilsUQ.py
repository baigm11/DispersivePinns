import torch
import numpy as np
import sobol_seq


def generator_points(type_of_coll, samples, dim, random_seed):
    if type_of_coll == "random":
        torch.random.manual_seed(random_seed)
        return torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_of_coll == "sobol":
        # if n_time_step is None:
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        return torch.from_numpy(data).type(torch.FloatTensor)


def convert(vector, extrema_values):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)