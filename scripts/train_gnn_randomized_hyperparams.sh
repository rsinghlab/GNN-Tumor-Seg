#!/bin/bash


#first activate your venv
export PATH="/anaconda3/bin:$PATH"
source activate gnn_tumor_seg

#set up your filepaths
#both the training and testing data are expected to have been processed with the preprocess_dataset script.
PROCESSED_TRAINING_DATA_DIR="/Users/camillo_stuff/singhlab/data/smallCIA/newppData"
LOG_OUTPUT_DIR="/Users/camillo_stuff/singhlab/GNN-Tumor-Seg/temp_output"
#I run this on a sun grid engine which i use to generate a unique name. Adapt according to use case
#MODEL_NAME="randomModel${SGE_TASK_ID}"
MODEL_NAME="randomModelTest"
NUM_FOLDS=3


python -m scripts.train_gnn -d $PROCESSED_TRAINING_DATA_DIR -o $LOG_OUTPUT_DIR -r "${MODEL_NAME}_gnn" -k $NUM_FOLDS -m GSgcn --random_hyperparams