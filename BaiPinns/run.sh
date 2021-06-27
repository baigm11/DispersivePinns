bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_15/Retrain_4 sobol 0.0 '{"hidden_layers": 4, "neurons": 20, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 95 false


# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_17/Retrain_0 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 10, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 42 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_16/Retrain_0 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 42 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_15/Retrain_0 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 42 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_17/Retrain_1 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 10, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 82 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_16/Retrain_1 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 82 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_15/Retrain_1 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 82 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_17/Retrain_2 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 10, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 15 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_16/Retrain_2 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 15 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_15/Retrain_2 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 15 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_17/Retrain_3 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 10, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 4 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_16/Retrain_3 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 4 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_15/Retrain_3 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 4 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_17/Retrain_4 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 10, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 95 false
#
# bsub -W 6:00 -R 'rusage[mem=8192,ngpus_excl_p=1]' -R 'select[gpu_model0==GeForceGTX1080Ti]' python3 PINNS2.py 0 16384 120 4096 0 0 None 0 0 2 EnsRadInv/Setup_15/Retrain_4 sobol 0.0 '{"hidden_layers": 8, "neurons": 24, "residual_parameter": 0.1, "kernel_regularizer": 2, "regularization_parameter": 0, "batch_size": 20600, "epochs": 1, "activation": "tanh"}' 95 false
