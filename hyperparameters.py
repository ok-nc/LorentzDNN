"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_LORENTZ = True
NUM_LORENTZ_OSC = 3
LINEAR = [8, 250, 250, 250]
# LINEAR = [2*NUM_LORENTZ_OSC, 12, 25, 50, 100, 200, 300]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-4
BATCH_SIZE = 512
EVAL_STEP = 10
RECORD_STEP = 10
TRAIN_STEP = 10000
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5
USE_CLIP = False
GRAD_CLIP = 5
USE_WARM_RESTART = True
LR_WARM_RESTART = 200
INT_LAYER_SIZE = 100
INT_LAYER_STR = 0.5

# Data Specific parameters
X_RANGE = [i for i in range(2, 10)]
Y_RANGE = [i for i in range(26, 2012)] # New frequency range: 0.8 - 1.21646
FREQ_LOW = 0.8
FREQ_HIGH = 1.21979
NUM_SPEC_POINTS = 300
FORCE_RUN = True
DATA_DIR = ''                # For local usage
# DATA_DIR = 'C:/Users/labuser/mlmOK_Pytorch/'                # For Omar office desktop usage
# DATA_DIR = '/home/omar/PycharmProjects/mlmOK_Pytorch/'  # For Omar laptop usage
GEOBOUNDARY =[30, 55, 42, 52]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None 
EVAL_MODEL = "Lorentz_good"
NUM_PLOT_COMPARE = 10
