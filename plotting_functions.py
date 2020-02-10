import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import evaluate_model
import seaborn as sns; sns.set()
import logging_functions
from sklearn.neighbors import NearestNeighbors
from pandas.plotting import table
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def RetrieveFeaturePredictionNMse(model_name):
    """
    Retrieve the Feature and Prediciton values and place in a np array
    :param model_name: the name of the model
    return Xtruth, Xpred, Ytruth, Ypred
    """
    # Retrieve the prediction and truth and prediction first
    feature_file = os.path.join('data', 'test_Xtruth_{}.csv'.format(model_name))
    pred_file = os.path.join('data', 'test_Ypred_{}.csv'.format(model_name))
    truth_file = os.path.join('data', 'test_Ytruth_{}.csv'.format(model_name))
    feat_file = os.path.join('data', 'test_Xpred_{}.csv'.format(model_name))

    # Getting the files from file name
    Xtruth = pd.read_csv(feature_file,header=None, delimiter=' ')
    Xpred = pd.read_csv(feat_file,header=None, delimiter=' ')
    Ytruth = pd.read_csv(truth_file,header=None, delimiter=' ')
    Ypred = pd.read_csv(pred_file,header=None, delimiter=' ')
    
    #retrieve mse, mae
    Ymae, Ymse = evaluate_model.compare_truth_pred(pred_file, truth_file) #get the maes of y
    
    print(Xtruth.shape)
    return Xtruth.values, Xpred.values, Ytruth.values, Ypred.values, Ymae, Ymse

def ImportColorBarLib():
    """
    Import some libraries that used in a colorbar plot
    """
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl
    print("import sucessful")
    
    return mpl
  
def UniqueMarkers():
    import itertools
    markers = itertools.cycle(( 'x','1','+', '.', '*','D','v','h'))
    return markers
  
def SpectrumComparisonNGeometryComparison(rownum, colnum, Figsize, model_name, boundary = [-1,1,-1,1]):
    """
    Read the Prediction files and plot the spectra comparison plots
    :param SubplotArray: 2x2 array indicating the arrangement of the subplots
    :param Figsize: the size of the figure
    :param Figname: the name of the figures to save
    :param model_name: model name (typically a list of numebr containing date and time)
    """
    mpl = ImportColorBarLib()    #import lib
    
    Xtruth, Xpred, Ytruth, Ypred, Ymae, Ymse =  RetrieveFeaturePredictionNMse(model_name)  #retrieve features
    print("Ymse shape:",Ymse.shape)
    print("Xpred shape:", Xpred.shape)
    print("Xtrth shape:", Xtruth.shape)
    #Plotting the spectrum comaprison
    f = plt.figure(figsize=Figsize)
    fignum = rownum * colnum
    for i in range(fignum):
      ax = plt.subplot(rownum, colnum, i+1)
      plt.ylabel('Transmission rate')
      plt.xlabel('frequency')
      plt.plot(Ytruth[i], label = 'Truth',linestyle = '--')
      plt.plot(Ypred[i], label = 'Prediction',linestyle = '-')
      plt.legend()
      plt.ylim([0,1])
    f.savefig('Spectrum Comparison_{}'.format(model_name))
    
    """
    Plotting the geometry comparsion, there are fignum points in each plot
    each representing a data point with a unique marker
    8 dimension therefore 4 plots, 2x2 arrangement
    
    """
    #for j in range(fignum):
    pointnum = fignum #change #fig to #points in comparison
    
    f = plt.figure(figsize = Figsize)
    ax0 = plt.gca()
    for i in range(4):
      truthmarkers = UniqueMarkers() #Get some unique markers
      predmarkers = UniqueMarkers() #Get some unique markers
      ax = plt.subplot(2, 2, i+1)
      #plt.xlim([29,56]) #setting the heights limit, abandoned because sometime can't see prediciton
      #plt.ylim([41,53]) #setting the radius limits
      for j in range(pointnum):
        #Since the colored scatter only takes 2+ arguments, plot 2 same points to circumvent this problem
        predArr = [[Xpred[j, i], Xpred[j, i]] ,[Xpred[j, i + 4], Xpred[j, i + 4]]]
        predC = [Ymse[j], Ymse[j]]
        truthplot = plt.scatter(Xtruth[j,i],Xtruth[j,i+4],label = 'Xtruth{}'.format(j),
                                marker = next(truthmarkers),c = 'm',s = 40)
        predplot  = plt.scatter(predArr[0],predArr[1],label = 'Xpred{}'.format(j),
                                c =predC ,cmap = 'jet',marker = next(predmarkers), s = 60)
      
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      rect = mpl.patches.Rectangle((boundary[0],boundary[2]),boundary[1] - boundary[0], boundary[3] - boundary[2],
																		linewidth=1,edgecolor='r',
                                   facecolor='none',linestyle = '--',label = 'data region')
      ax.add_patch(rect)
      plt.autoscale()
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                 mode="expand",ncol = 6, prop={'size': 5})#, bbox_to_anchor=(1,0.5))
    
    cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = f.colorbar(predplot, cax=cb_ax)
    #f.colorbar(predplot)
    f.savefig('Geometry Comparison_{}'.format(model_name))


class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """
    def __init__(self, bv_loss, f1, f2 = None, f1_name = 'f1', f2_name = 'f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        self.f1_name = f1_name
        self.f2_name = f2_name
        #print(type(f1))
    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }


def HeatMapBVL(plot_x_name, plot_y_name, title,  save_name='HeatMap.png', HeatMap_dir = 'HeatMap',
                feature_1_name=None, feature_2_name=None,
                heat_value_name = 'best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hyperswiping thing
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False          #indication flag of whether it is a 1d or 2d plot to plot
    #Check the data integrity 
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    #Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    df_list = []                        #make a list of data frame for further use
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
             if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name) #Get the file relative path from 
                # df = pd.read_csv(file_path, index_col=0)
                flag = logging_functions.load_flags(subdir)
                flag_dict = vars(flag)
                df = pd.DataFrame()
                for k in flag_dict:
                    df[k] = pd.Series(str(flag_dict[k]), index=[0])
                print(df)
                if (one_dimension_flag):
                    #print(df[[heat_value_name, feature_1_name]])
                    #print(df[heat_value_name][0])
                    #print(df[heat_value_name].iloc[0])
                    df_list.append(df[[heat_value_name, feature_1_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])), 
                                                f1_name = feature_1_name))
                else:
                    if feature_2_name == 'linear_unit':                         # If comparing different linear units
                        df['linear_unit'] = eval(df[feature_1_name][0])[1]
                        df['best_validation_loss'] = get_bvl(file_path)
                    df_list.append(df[[heat_value_name, feature_1_name, feature_2_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]),eval(str(df[feature_1_name][0])),
                                                eval(str(df[feature_2_name][0])), feature_1_name, feature_2_name))
    
    print(df_list)
    #Concatenate all the dfs into a single aggregate one for 2 dimensional usee
    df_aggregate = pd.concat(df_list, ignore_index = True, sort = False)
    #print(df_aggregate[heat_value_name])
    #print(type(df_aggregate[heat_value_name]))
    df_aggregate.astype({heat_value_name: 'float'})
    #print(type(df_aggregate[heat_value_name]))
    #df_aggregate = df_aggregate.reset_index()
    print("before transformation:", df_aggregate)
    [h, w] = df_aggregate.shape
    for i in range(h):
        for j in range(w):
            if isinstance(df_aggregate.iloc[i,j], str) and (isinstance(eval(df_aggregate.iloc[i,j]), list)):
                # print("This is a list!")
                df_aggregate.iloc[i,j] = len(eval(df_aggregate.iloc[i,j]))

    print("after transoformation:",df_aggregate)
    
    #Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt, 
                                                                point.bv_loss, point.feature_1, point.feature_2))
        assert(isinstance(point.bv_loss, float))        #make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    
    f = plt.figure()
    #After we get the full list of HMpoint object, we can start drawing 
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key = lambda x: x.feature_1)
        #Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list_sorted:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:",feature_1_list)
        #start plotting
        plt.plot(feature_1_list, bv_loss_list,'o-')
    else: #Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        #point_df = pd.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        df_aggregate = df_aggregate.reset_index()
        df_aggregate.sort_values(feature_1_name, axis=0, inplace=True)
        df_aggregate.sort_values(feature_2_name, axis=0, inplace=True)
        df_aggregate.sort_values(heat_value_name, axis=0, inplace=True)
        print("before dropping", df_aggregate)
        df_aggregate = df_aggregate.drop_duplicates(subset=[feature_1_name, feature_2_name], keep='first')
        print("after dropping", df_aggregate)
        point_df_pivot = df_aggregate.reset_index().pivot(index=feature_1_name, columns=feature_2_name, values=heat_value_name).astype(float)
        point_df_pivot = point_df_pivot.rename({'5': '05'}, axis=1)
        point_df_pivot = point_df_pivot.reindex(sorted(point_df_pivot.columns), axis=1)
        print("pivot=")
        csvname = HeatMap_dir + 'pivoted.csv'
        point_df_pivot.to_csv(csvname)
        print(point_df_pivot)
        sns.heatmap(point_df_pivot, cmap = "YlGnBu")
    plt.xlabel(plot_y_name)                 # Note that the pivot gives reversing labels
    plt.ylabel(plot_x_name)                 # Note that the pivot gives reversing labels
    plt.title(title)
    plt.savefig(save_name)


def PlotPossibleGeoSpace(figname, Xpred_dir, compare_original = False,calculate_diversity = None):
    """
    Function to plot the possible geometry space for a model evaluation result.
    It reads from Xpred_dir folder and finds the Xpred result insdie and plot that result
    :params figname: The name of the figure to save
    :params Xpred_dir: The directory to look for Xpred file which is the source of plotting
    :output A plot containing 4 subplots showing the 8 geomoetry dimensions
    """
    Xpredfile = logging_functions.get_Xpred(Xpred_dir)
    Xpred = pd.read_csv(Xpredfile, header=None, delimiter=' ').values
    
    Xtruthfile = logging_functions.get_Xtruth(Xpred_dir)
    Xtruth = pd.read_csv(Xtruthfile, header=None, delimiter=' ').values

    f = plt.figure()
    ax0 = plt.gca()
    print(np.shape(Xpred))
    #print(Xpred)
    #plt.title(figname)
    if (calculate_diversity == 'MST'):
        diversity_Xpred, diversity_Xtruth = calculate_MST(Xpred, Xtruth)
    elif (calculate_diversity == 'AREA'):
        diversity_Xpred, diversity_Xtruth = calculate_AREA(Xpred, Xtruth)

    for i in range(4):
      ax = plt.subplot(2, 2, i+1)
      ax.scatter(Xpred[:,i], Xpred[:,i + 4],s = 3,label = "Xpred")
      if (compare_original):
          ax.scatter(Xtruth[:,i], Xtruth[:,i+4],s = 3, label = "Xtruth")
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      plt.xlim(-1,1)
      plt.ylim(-1,1)
      plt.legend()
    if (calculate_diversity != None):
        plt.text(-4, 3.5,'Div_Xpred = {}, Div_Xtruth = {}, under criteria {}'.format(diversity_Xpred, diversity_Xtruth, calculate_diversity), zorder = 1)
    plt.suptitle(figname)
    f.savefig(figname+'.png')

def PlotPairwiseGeometry(figname, Xpred_dir):
    """
    Function to plot the pair-wise scattering plot of the geometery file to show
    the correlation between the geometry that the network learns
    """
    
    Xpredfile = logging_functions.get_Xpred(Xpred_dir)
    Xpred = pd.read_csv(Xpredfile, header=None, delimiter=' ')
    f=plt.figure()
    axes = pd.plotting.scatter_matrix(Xpred, alpha = 0.2)
    #plt.tight_layout()
    plt.title("Pair-wise scattering of Geometery predictions")
    plt.savefig(figname)

def calculate_AREA(Xpred, Xtruth):
    """
    Function to calculate the area for both Xpred and Xtruth under using the segmentation of 0.01
    """
    area_list = np.zeros([2,4])
    X_list = [Xpred, Xtruth]
    binwidth = 0.05
    for cnt, X in enumerate(X_list):
        for i in range(4):
            hist, xedges, yedges = np.histogram2d(X[:,i],X[:,i+4], bins = np.arange(-1,1+binwidth,binwidth))
            area_list[cnt, i] = np.mean(hist > 0)
    X_histgt0 = np.mean(area_list, axis = 1)
    assert len(X_histgt0) == 2
    return X_histgt0[0], X_histgt0[1]

def calculate_MST(Xpred, Xtruth):
    """
    Function to calculate the MST for both Xpred and Xtruth under using the segmentation of 0.01
    """

    MST_list = np.zeros([2,4])
    X_list = [Xpred, Xtruth]
    for cnt, X in enumerate(X_list):
        for i in range(4):
            points = X[:,i:i+5:4]
            distance_matrix_points = distance_matrix(points,points, p = 2)
            csr_mat = csr_matrix(distance_matrix_points)
            Tree = minimum_spanning_tree(csr_mat)
            MST_list[cnt,i] = np.sum(Tree.toarray().astype(float))
    X_MST = np.mean(MST_list, axis = 1)
    return X_MST[0], X_MST[1]


def get_bvl(file_path):
    """
    This is a helper function for 0119 usage where the bvl is not recorded in the pickled object but in .txt file and needs this funciton to retrieve it
    """
    df = pd.read_csv(file_path, delimiter=',')
    bvl = 0
    for col in df:
        if 'best_validation_loss' in col:
            print(col)
            strlist = col.split(':')
            bvl = eval(strlist[1][1:-2])
    if bvl == 0:
        print("Error! We did not found a bvl in .txt.file")
    else:
        return float(bvl)

