#!/bin/bash

#requires you to have saved logits from a GNN model

#first activate your venv
export PATH="/anaconda3/bin:$PATH"
source activate brain_tumor_segmentation

#set up your filepaths
#both the training and testing data are expected to have been processed with the preprocess_dataset script.
PROCESSED_TRAINING_DATA_DIR="/Users/camillo_stuff/singhlab/data/smallCIA/newppData"
LOG_OUTPUT_DIR="/Users/camillo_stuff/singhlab/GNN-Tumor-Seg/temp_output"
GNN_LOGIT_DIR="/Users/camillo_stuff/singhlab/GNN-Tumor-Seg/temp_logit"
#I run this on a sun grid engine which i use to generate a unique name. Adapt according to use case
#MODEL_NAME="randomModel${SGE_TASK_ID}"
MODEL_NAME="randomModelTest"
NUM_FOLDS=3


python -m scripts.train_refinement_cnn -d $PROCESSED_TRAINING_DATA_DIR  -o $LOG_OUTPUT_DIR -l $GNN_LOGIT_DIR -r "${MODEL_NAME}_cnn" -k $NUM_FOLDS