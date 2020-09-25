"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_LORENTZ = True
NUM_LORENTZ_OSC = 2
LINEAR = [2*NUM_LORENTZ_OSC, 50, 100, 200, 300]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 2048
EVAL_STEP = 10
RECORD_STEP = 20
TRAIN_STEP = 10000
LEARN_RATE = 1e-2
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5
USE_CLIP = False
GRAD_CLIP = 5
USE_WARM_RESTART = True
LR_WARM_RESTART = 200

# Data Specific parameters
X_RANGE = [i for i in range(0, 2*NUM_LORENTZ_OSC)]
Y_RANGE = [i for i in range(2*NUM_LORENTZ_OSC, 300+2*NUM_LORENTZ_OSC)]
FREQ_LOW = 0.5
FREQ_HIGH = 5
NUM_SPEC_POINTS = 300
FORCE_RUN = True
# DATA_DIR = ''                # For local usage
# DATA_DIR = 'C:/Users/labuser/mlmOK_Pytorch/'                # For Omar office desktop usage
DATA_DIR = '/home/omar/PycharmProjects/mlmOK_Pytorch/'  # For Omar laptop usage
GEOBOUNDARY =[20, 200, 20, 100]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None 
EVAL_MODEL = "20200820_114052-HL-100-100-100"
NUM_PLOT_COMPARE = 10
