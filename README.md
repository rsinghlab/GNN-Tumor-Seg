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

Requests

# Overview

We first preprocess the data provided by the competition into the format used by our model. Specifically, the script preprocess.py accomplishes the following:

    1. Normalize and standardize each image of each MRI modality

    2. Combine multiple MRI modalitities into one image array

    3. Swap labels from BraTS order (0,2,1,4) to more intuitive order (0,1,2,3)

    4. Convert image into a graph. This is done by running a supervoxel creation algorithm on the standardized and combined image and converting each supervoxel into a graph node, with edges between neighboring supervoxels.

The converted data is then stored in a seperate directory, which is used as input for subsequent scripts, e.g. training a model.

Example: "python -m scripts.preprocess_dataset -d ~/project_data/BraTS21_data/raw/train -n 15000 -k 0 -b 0.5 -o ~/project_data/BraTS21_data/processed/train -l _seg.nii.gz -p BraTS2021"

The CLI arguments are explained in the script.



modify hyperparams in hyperparam helpers

modify filepaths (optional)

modify filepaths in bash script