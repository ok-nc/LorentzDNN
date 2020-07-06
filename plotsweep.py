import torch
import sys
import plotting_functions
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['hypersweep_oops']
    for pathname in pathnamelist:
        plotting_functions.HeatMapBVL('Delta_0','Lr_0','Delta vs Learning Rate Heat Map',
                                      save_name=pathname + '_heatmap.png',
                                HeatMap_dir='models/'+pathname,feature_1_name='delta',feature_2_name='lr')
