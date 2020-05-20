"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import flagreader
from network_wrapper import Network
from network_model import Forward
import datareader
from plotting_functions import plotMSELossDistrib_eval
# Libs

def evaluate_from_model(model_dir):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print("Retrieving flag object for parameters")
    flags = flagreader.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode

    # Get the data
    # train_loader, test_loader = datareader.read_data(flags)
    train_loader, test_loader = datareader.read_data(x_range=flags.x_range,
                                                     y_range=flags.y_range,
                                                     geoboundary=flags.geoboundary,
                                                     batch_size=flags.batch_size,
                                                     normalize_input=flags.normalize_input,
                                                     data_dir=flags.data_dir,
                                                     test_ratio=0.999)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    plotMSELossDistrib_eval(pred_file, truth_file, flags)
    print("Evaluation finished")


def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None

if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flagreader.read_flag()

    print(useless_flags.eval_model)
    # Call the evaluate function from model
    evaluate_from_model(useless_flags.eval_model)