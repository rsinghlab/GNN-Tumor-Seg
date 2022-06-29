#!/bin/bash

#Example of how to run the train_cnn script with randomized hyperparameters.
#Prior to running, make sure you have both preprocessed data and generated logits from a GNN model.

#activate virtual environment
export PATH="/anaconda3/bin:$PATH"
source activate brain_tumor_segmentation


PROCESSED_TRAINING_DATA_DIR="~/project_data/BraTS21_data/processed/train"
LOG_OUTPUT_DIR="~/code/GNN-Tumor-Seg/logs"
GNN_LOGIT_DIR="~/project_data/BraTS21_data/logits/train"
MODEL_NAME="randomModelTest"
NUM_FOLDS=2


python -m scripts.train_refinement_cnn -d $PROCESSED_TRAINING_DATA_DIR  -o $LOG_OUTPUT_DIR -l $GNN_LOGIT_DIR -r "${MODEL_NAME}_cnn" -k $NUM_FOLDS