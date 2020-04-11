"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_LORENTZ = True
NUM_LORENTZ_OSC = 4
USE_CONV = False                         # Whether use upconv layer when not using lorentz @Omar
LINEAR = [8, 100, 100]
# If the Lorentzian is False
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 1024
EVAL_STEP = 10
RECORD_STEP = 10
TRAIN_STEP = 1000
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5
USE_CLIP = True
GRAD_CLIP = 10
USE_WARM_RESTART = False
LR_WARM_RESTART = 300
ERR_EXP = 4

# Data Specific parameters
X_RANGE = [i for i in range(0, 8 )]
Y_RANGE = [i for i in range(8 , 308 )]
FREQ_LOW = 0.5
FREQ_HIGH = 5
FORCE_RUN = True
DATA_DIR = ''                # For local usage
# DATA_DIR = 'C:/Users/labuser/mlmOK_Pytorch/'                # For Omar office desktop usage
# DATA_DIR = '/home/omar/PycharmProjects/mlmOK_Pytorch/'  # For Omar laptop usage
GEOBOUNDARY =[20, 200, 20, 100]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None 
EVAL_MODEL = "20200217_112744"
NUM_PLOT_COMPARE = 10
