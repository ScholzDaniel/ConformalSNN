# Capturing Uncertainty Over Time for Spiking Neural Networks by Exploiting Conformal Prediction Sets

This repository is the official implementation of our paper with ID 197.

## 📋 Requirements

First, a python environment needs to be set up. Conda is not needed but
recommended. We used python 3.9.

`conda create -n cp-snn python=3.9`

Activate the environment and install all requirements.

`conda activate cp-snn`

`pip install -r requirements.txt`

Git LFS is used to store larger files, make sure it is available on your system.
Initialize once via

`git lfs install`


## 👷 Project Files
All python files with the prefix "run_" are the main files to be run.
Flags are not mandatory but are available when one wants to change the default
as specified below.

## 📋 Training Networks

File run_train.py includes the training procedure for both Evidential Deep
Learning
(EDL) and Conformal Prediction (CP) approaches. File run_train_ese.py includes
the training procedure for the Ensemble (ESE - referred to as NNE in our paper)
approach.

The training is computational expensive, we used an NVIDIA RTX 3090 GPU and 64
GB of RAM.

All trainings yield Spiking Neural Networks computed stateful in fully-connected
feed-forward fashion where per step one time-bin (frame) is fed through the
network.

To train a network either for CP or EDL experiments (CP specified below):

`python run_train.py --train_mode CP --model Soli --hidden_units 100`

To train multiple networks (5 specified below) and save the models such that ESE
experiments can be run:

`python run_train_ese.py --model Soli --hidden_units 100 --runs 5`

## 📋 Running Experiments

Files run_cp.py, run_edl.py, and run_ese.py include all our experiments for CP,
EDL, and ESE, respectively.

Reproduce Figure 4 and Figure 5 and print values of Table 2 for the Soli
dataset:

`python run_cp.py --model Soli --allow_empty false`

Print values of Table 3 for the Soli dataset:

`python run_cp.py --model Soli --allow_empty true`

Omit the --allow_empty flag for EDL and ESE experiments.
