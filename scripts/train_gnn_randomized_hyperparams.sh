#!/bin/bash

#Example of how to run the train_gnn script with randomized hyperparameters.
#both the training and testing data are expected to have been processed with the preprocess_dataset script.

#activate virtual environment
export PATH="/anaconda3/bin:$PATH"
source activate gnn_tumor_seg

#set up your filepaths
#both the training and testing data are expected to have been processed with the preprocess_dataset script.
PROCESSED_TRAINING_DATA_DIR="~/project_data/BraTS21_data/processed/train"
LOG_OUTPUT_DIR="~/code/GNN-Tumor-Seg/logs"
MODEL_NAME="randomModelTest"
NUM_FOLDS=3


python -m scripts.train_gnn -d $PROCESSED_TRAINING_DATA_DIR -o $LOG_OUTPUT_DIR -r "${MODEL_NAME}_gnn" -k $NUM_FOLDS -m GSgcn --random_hyperparams