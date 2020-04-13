# Machine Learning with Metamaterials 
## Author: Omar Khatib

## This is a machine learning network designed to aid in the modeling and simulation of metamaterials (MMs). 
### The original forward network model was developed by Christian Nadell and Bohao Huang in tensoflow, and further optimized and ported to pytorch by Ben Ren. 

### The main aim of this project is to incorporate known physics relations to improve deep learning networks applied to MM systems. 
### The fundamental relationships involve using Lorentz oscillator equations as the basis for mapping an input geometry to output transmission, reflection, or absorption spectra. 

# Developer Log:

### Roadmap for this work:
1. Incorporate Lorentz oscillator equations into neural network
2. Train a toy model to learn prescribed relationship of lorentz osc parameters to input geometry
3. Train network on CST simulations for 2x2 unit cell MMs using a Lorentz layer. 
4. Incorporate auxilliary network(s) for capturing 2nd order effects (e.g. scattering, spatial dispersion, etc)

## < 2020.02.07
All effort up to now has attempted to use previous network architecture + final Lorentz layer, with limited success.
This includes input layer (8), linear layers (25-1000), and lorentz layer (12-150), with Relu activation for linear layers, and sigmoid for Lorentz layer. 
Can achieve MSE ~ 5e-3, but fits do not look good. Need better architecture/strategy to find global min of loss surface.
Old forward model (linear + conv layers) can achieve ~ 6-8e-4 MSE without Lorentz layer.  

## 2020.03.6
Model successfully trains on e2 spectra with pretraining on Lorentz parameters. Exploding/vanishing gradients
are a big problem. Fixed by either clipping or using SmoothL1MSEloss to contain gradients. Model can train on e2 spectra
without pretraining, but only if last linear layer uses sigmoid activation + Lorentz parameter bounds 
(5,5,0.5 for toy model). Best performance ~ 0.05-0.08 MSE with pretty good looking oscillator fits. 

## 2020.03.10
Trying to remove constraints on model training on e2 sim data. Replacing sigmoid with relu (except for damping param)
results in good but not great fits (0.6-1 MSE). Has trouble fitting smaller/narrower oscillators, as loss tends
to get dominated by big/strong ones, and model plateaus nearest local min. 

## To dos:

Short term
- give each lorentz parameter tensor its own fc network with bn
- implement a peak-searching loss term (using derivative)
- update/fix model evaluation/inference mode (default to not saving csv files)
- fix tensorboard (add external script that takes latest folder and updates batch file)


Longer term
1. Organize plotting and analysis functions into separate module
2. Find best way to initialize params + best loss function (physics constrained)
3. Active/transfer learning
4. Lorentz activation functions


