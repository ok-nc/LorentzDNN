# Machine Learning with Metamaterials 
## Author: Omar Khatib

## This is a machine learning network designed to aid in the modeling and simulation of metamaterials (MMs). 
## The original forward network model was developed by Christian Nadell and Bohao Huang in tensoflow, and further optimized and ported to pytorch by Ben Ren. 

### The main aim of this project is to incorporate known physics relations to improve deep learning applied to MM systems. 
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

## To dos:
1. Find best way to initialize params + best optimizer. 
2. Pretrain on single spectrum
3. Lorentz activation functions


