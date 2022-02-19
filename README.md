# GNN-Tumor-Seg
Using Graph Neural Networks to Segment MRIs of Brain Tumors.

This repository contains the code used in our publication xxxx. This model was submitted to the BraTS2021 competition.


# Data Access
The training and validation data for Task 1 of the BraTS2021 competition can be accessed through Synapse (https://www.synapse.org/#!Synapse:syn25829067/wiki/610865). More information on the BraTS competition can be found at http://www.braintumorsegmentation.org.


# Requirements
Numpy>=1.17

Scipy>=1.4

Scikit-image

Networkx

Nibabel

Pytorch>=1.7

DGL>=0.4

# Overview

We first preprocess the data provided by the competition into the format used by our model.

This can be done with the script preprocess_data.py.

Example: "python -m scripts.preprocess_dataset -d ~/project_data/BraTS21_data/raw/train -n 15000 -k 0 -b 0.5 -o ~/project_data/BraTS21_data/processed/train -l _seg.nii.gz -p BraTS2021"

The CLI arguments are explained in the script.



modify hyperparams in hyperparam helpers

modify filepaths (optional)

modify filepaths in bash script