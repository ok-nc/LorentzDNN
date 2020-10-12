import torch
import sys
import plotting_functions
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    # pathnamelist = ['hypersweep_oops']
    savepath = 'C:/Users/labuser/mlmOK_Pytorch/models/hypersweep_regscale_num_osc/'
    # for pathname in savepath:
    plotting_functions.HeatMapBVL('Reg scale','Num Osc','Reg scale vs Num Osc Heat Map',
                                  save_name=savepath + 'heatmap.png',
                            HeatMap_dir=savepath,feature_1_name='reg_scale',feature_2_name='num_lorentz_osc')
