import torch
import sys
import plotting_functions
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    # pathnamelist = ['hypersweep_oops']
    savepath = 'C:/Users/labuser/mlmOK_Pytorch/models/hs_int_layer/'
    # for pathname in savepath:
    plotting_functions.HeatMapBVL('Int Strength','Int size','Layer size vs Layer Num',
                                  save_name=savepath + 'heatmap.png',
                            HeatMap_dir=savepath,feature_1_name='int_layer_str',feature_2_name='int_layer_size')
