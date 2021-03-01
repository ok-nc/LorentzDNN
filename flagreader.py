"""
This file serves to hold helper functions that is related to the "Flag" object which contains
all the parameters during training and inference
"""
# Built-in

import argparse
import pickle
import os
# Libs

# Own module
from hyperparameters import *

# Torch

def read_flag():
    """
    This function is to write the read the flags from a parameter file and put them in formats
    :return: flags: a struct where all the input params are stored
    """

    parser = argparse.ArgumentParser()

    # Model Architectural Params
    parser.add_argument('--use-lorentz', type=bool, default=USE_LORENTZ,
                        help='The boolean flag that indicate whether we use lorentz oscillators')
    parser.add_argument('--num-lorentz-osc', type=int, default=NUM_LORENTZ_OSC,
                        help='Number of lorentz oscillators to use')
    parser.add_argument('--linear', type=list, default=LINEAR, help='The fc layers units')

    # Optimization params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--record-step', default=RECORD_STEP, type=int, help='# steps between recording images to tb')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
#    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
#                        help='decay learning rate at this number of steps')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    parser.add_argument('--grad_clip', default=GRAD_CLIP, type=float,
                        help='Gradient clipping value')
    parser.add_argument('--use_clip', default=USE_CLIP, type=bool,
                        help='Gradient clipping toggle')
    parser.add_argument('--use-warm-restart', default=USE_WARM_RESTART, type=bool,
                        help='Use warm restart learning rate scheduling')
    parser.add_argument('--lr-warm-restart', default=LR_WARM_RESTART, type=int, help='# steps before warm restart')
    parser.add_argument('--int-layer-size', default=INT_LAYER_SIZE, type=int, help='size of input interaction layer')
    parser.add_argument('--int-layer-str', default=INT_LAYER_STR, type=float, help='input coupling strength')


    # Data specific Params
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--freq-low', default=FREQ_LOW, type=float, help='lower bound to frequency range')
    parser.add_argument('--freq-high', default=FREQ_HIGH, type=float, help='upper bound to frequency range')
    parser.add_argument('--num-spec-points', type=int, default=NUM_SPEC_POINTS,
                        help='Number of spectral points in output')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool,
                        help='whether we should normalize the input or not')
    parser.add_argument('--test-ratio', default=TEST_RATIO, type=float, help='the ratio of test case')

    # Running specific
    parser.add_argument('--eval-model', default=EVAL_MODEL, type=str,
                        help='the folder name of the model that you want to evaluate')
    parser.add_argument('--use-cpu-only', type=bool, default=USE_CPU_ONLY,
                        help='The boolean flag that indicate use CPU only')
    parser.add_argument('--num-plot-compare', type=int, default=NUM_PLOT_COMPARE,
                        help='#Plots to store in tensorboard during training for spectra compare')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')

    # flags = parser.parse_args()  # This is for command line version of the code
    flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags


def save_flags(flags, save_file="flags.obj"):
    """
    This function serialize the flag object and save it for further retrieval during inference time
    :param flags: The flags object to save
    :param save_file: The place to save the file
    :return: None
    """
    with open(save_file,'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object


def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags

def write_flags_and_BVE(flags, best_validation_loss):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :return: None
    """
    #To avoid terrible looking shape of y_range
    yrange = flags.y_range
    # yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    yrange_str = [yrange[0], yrange[-1]]
    flags_dict = vars(flags)
    flags_dict_copy = flags_dict.copy()                 # in order to not corrupt the original data structure
    flags_dict_copy['y_range'] = yrange_str             # Change the y range to be acceptable long string
    flags_dict_copy['best_validation_loss'] = best_validation_loss #Append the bvl
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    print(flags_dict_copy)
    with open('parameters.txt','w') as f:
        print(flags_dict_copy, file = f )
    # Pickle the obj
    save_flags(flags)

