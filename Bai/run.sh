#Kawa smaller soliton in double soliton
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 2048 1024 0 0 None Test Kawa_double_small1 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 3072, "epochs": 1, "max_iter": 50000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_double_small2 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12278, "epochs": 1, "max_iter": 50000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_double_small3 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 50000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#Kawa double soliton
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_double39 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_double40 sobol 0.0 '{"hidden_layers": 5, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_double41 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_double42 sobol 0.0 '{"hidden_layers": 5, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_double22 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

#large train number
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 25000 10000 0 0 None Test Kawa_double31 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 35000, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#Kawa double soliton case2
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_d27 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_d18 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_d28 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_d20 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#Kawa triple soliton
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 4096 2048 0 0 None Test Kawa_t1 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 6144, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_t2 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_t3 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#Kawa anti gen
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 4096 2048 0 0 None Test Kawa_agen7 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 6144, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test Kawa_agen8 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test Kawa_agen9 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false



#CH double peakon
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 2048 1024 0 0 None Test CH_anti1 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 3072, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 4096 2048 0 0 None Test CH_anti2 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 6144, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test CH_anti3 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test CH_anti4 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#CH single peakon lim
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 4096 2048 0 0 None Test CH_single_lim1 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 6144, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test CH_single_lim2 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test CH_single_lim3 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#CH double peakon lim
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 4096 2048 0 0 None Test CH_double_lim1 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 6144, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test CH_double_lim2 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test CH_double_lim4 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#BO single peri soliton
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 6615 5000 0 0 None Test BO_single23 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 11615, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 6615 5000 0 0 None Test BO_single24 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 11615, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 10935 8000 0 0 None Test BO_single25 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 18935, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 10935 8000 0 0 None Test BO_single26 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 18935, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16335 10000 0 0 None Test BO_single27 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 26335, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16335 10000 0 0 None Test BO_single28 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 26335, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 22815 12000 0 0 None Test BO_single29 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 34815, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 22815 12000 0 0 None Test BO_single30 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 34815, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false


#BO double line soliton
bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test BO_double50 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 8192 4096 0 0 None Test BO_double_peri2 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 12288, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test BO_double51 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 8192 0 0 None Test BO_double_peri4 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 24576, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 23000 15000 0 0 None Test BO_double52 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 38000, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 23000 15000 0 0 None Test BO_double_peri6 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 38000, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 32768 16384 0 0 None Test BO_double53 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 49152, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 90000 30000 0 0 None Test BO_double54 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 120000, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false

# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 32768 16384 0 0 None Test BO_double_peri8 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 49152, "epochs": 1, "max_iter": 100000, "activation": "tanh", "optimizer": "LBFGS"}' 32 false













