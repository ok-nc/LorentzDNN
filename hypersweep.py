"""
This .py file is to run train_network.py for hyper-parameter swipping in a linear fashion.
"""
import train_network
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flagreader
if __name__ == '__main__':
    # Setting the loop for setting the parameter
     for learning_rate in [1e-2, 1e-3, 1e-4]:
         for size in [12,50,100,200,500]:
            for i in range(3,9):
                flags = flagreader.read_flag()  	#setting the base case
                linear = [size for j in range(i)]        #Set the linear units
                linear[0] = 8                   # The start of linear
                linear[-1] = 12                # The end of linear
                flags.linear = linear
                flags.learn_rate = learning_rate
                # for j in range(3):
                #     flags.model_name = "trial_"+str(j)+"_complexity_swipe_layer_num" + str(i)
                #     train.training_from_flag(flags)
                flags.model_name = "size_" + str(size) + "_num_hidden_layers_" + str(i) + "_learning_rate_" + str(learning_rate)
                train_network.training_from_flag(flags)
