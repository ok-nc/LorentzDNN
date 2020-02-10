"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_LORENTZ = False
USE_CONV = True                         # Whether use upconv layer when not using lorentz @Omar
LINEAR = [8, 150, 150, 150, 150]
FIX_W0 = False
# If the Lorentzian is Flase
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 512
EVAL_STEP = 20
TRAIN_STEP = 1000
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-4

# Data Specific parameters
X_RANGE = [i for i in range(0, 8 )]
Y_RANGE = [i for i in range(8 , 308 )]
FORCE_RUN = True
DATA_DIR = '../'                # For local usage
DATA_DIR = '/home/omar/PycharmProjects/mlmOK_Pytorch/'                # For Omar laptop usage
GEOBOUNDARY =[20, 200, 20, 100]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None 
EVAL_MODEL = "20191202_161923"
NUM_COM_PLOT_TENSORBOARD = 1
