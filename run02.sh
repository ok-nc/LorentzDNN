#!/bin/bash
#SBATCH --output output02.err                                                   # output log file
#SBATCH -e error02.err                                                   # error log file
#SBATCH --mem=20G                                                      # request 20G memory
#SBATCH -c 1                                                           # request 6 gpu cores                                    
#SBATCH -p collinslab                                     # request 1 gpu for this job
#SBATCH --exclude=dcc-collinslab-gpu-[01,03,04]
module load Anaconda3/3.5.2                                            # load conda to make sure use GPU version of tf
# add cuda and cudnn path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
# add my library path
export PYTHONPATH=$PYTHONPATH:/hpc/home/sr365/Pytorch
# execute my file
# python hyperswipe02.py
python train_network.py
