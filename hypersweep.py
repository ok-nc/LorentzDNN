"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import os
import time
import flagreader
if __name__ == '__main__':
    # Setting the loop for setting the parameter
    #  for learning_rate in [1e-2, 1e-3, 1e-4]:
    model_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    for int_size in [250, 500, 1000, 1500]:
        for int_str in [0.01, 0.05, 0.1, 0.5]:
            flags = flagreader.read_flag()  # setting the base case
            flags.int_layer_size = int_size
            flags.int_layer_str = int_str
            for j in range(1):
                flags.model_name = model_name + "_intSize" + str(int_size) + "_intStr" + str(int_str) \
                                   + "_trial_" + str(j)
                train.training_from_flag(flags)

    # for num_osc in [10,15,20,25]:
    #     for layer_size in [25,100,200,250]:
    #         lin1 = [8, layer_size, layer_size]
    #         lin2 = [8, layer_size, layer_size, layer_size]
    #         lin3 = [8, layer_size, layer_size, layer_size, layer_size]
    #         # lin4 = [8, layer_size, layer_size, layer_size, layer_size, layer_size]
    #
    #         for layer_num in [lin1, lin2, lin3]:
    #             flags = flagreader.read_flag()  	#setting the base case
    #             flags.linear = layer_num
    #             flags.num_lorentz_osc = num_osc
    #             # flags.learn_rate = learning_rate
    #             for j in range(1):
    #                 flags.model_name = model_name + "_numOsc" + str(num_osc) + "_layerSize" + str(layer_size) \
    #                                    + "_layerNum" + str(len(layer_num)-1) + "_trial_"+str(j)
    #                 train.training_from_flag(flags)
                # flags.model_name = "size_" + str(size) + "_num_hidden_layers_" + str(i)
                # flags.model_name = "size_" + str(size) + "_num_hidden_layers_" + str(i) + "_learning_rate_" + str(learning_rate)
                # flags.model_name = model_name + "_reg" + str(regscale) + "_batch" + str(batchsize) + "num_osc" + str(num_osc)
                # train.training_from_flag(flags)

# if __name__ == '__main__':
#
#     flags = flagreader.read_flag()  # setting the base case
#     # flags.linear = [8, 100, 100, 12]
#     model_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     # model_description = "Smooth_L1_Loss_Warm_Restart"
#     # model_description = "MSE_Loss_Grad_Clip"
#     model_description = "delta_"
#     # for restart in [200, 500]:
#     #     for exp in [4,8]:
#     #         for clip in [20]:
#     #             # flags.lr_warm_restart = restart
#     #             # flags.use_warm_restart = True
#     #             flags.grad_clip = clip
#     #             for i in range(5):
#     #                 flags.linear = [8, 100, 100, 12]
#     #                 flags.model_name = model_name + model_description +str(exp)  + '_WRst_' + str(restart) + "_GC_" + \
#     #                                    str(clip) + "_run" + str(i + 1)
#     #                 # flags.model_name = model_name + model_description + "_L" + str(exp) +"_GC_" + \
#     #                 #                    str(clip) + "_run" + str(i + 1)
#     #                 train_network.training_from_flag(flags)
#     # for i in range(3):
#     #     flags.linear = [8, 100, 100, 100]
#     #     flags.model_name = model_name + '_' + model_description + "_run" + str(i + 1)
#     #     train.training_from_flag(flags)
#     for delta_0 in [0.5, 0.1, 0.05, 0.01]:
#         for lr_0 in [1e-4, 1e-3, 1e-2, 1e-1]:
#             for i in range(3):
#                 # flags.linear = [8, 30, 30]
#                 flags.lr = lr_0
#                 flags.delta = delta_0
#                 flags.model_name = model_name + '_' + model_description + str(delta_0) + '_lr_' + str(lr_0) +"_run"+str(i)
#                 train.training_from_flag(flags)