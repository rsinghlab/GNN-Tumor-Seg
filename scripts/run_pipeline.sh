#!/bin/bash

#This script can be used to train a GNN and CNN on a training dataset and then produce predictions on a test dataset.
#Trains one model using the hyperparameters specified in the hyperparameter_helpers file as well as the prediction scripts.

#first activate your venv
export PATH="/anaconda3/bin:$PATH"
source activate gnn_tumor_seg

#set up your filepaths
#both the training and testing data are expected to have been processed with the preprocess_dataset script.
PROCESSED_TRAINING_DATA_DIR="~/project_data/BraTS21_data/processed/train"
PROCESSED_TEST_DATA_DIR="~/project_data/BraTS21_data/processed/val"
GNN_LOGIT_DIR="~/project_data/BraTS21_data/logits/train"
PRED_OUTPUT_DIR="~/project_data/BraTS21_data/preds/val"
LOG_OUTPUT_DIR="~/code/GNN-Tumor-Seg/logs"
MODEL_NAME="bashModel"


echo "Starting step 1 of 4: Training GNN"
#python -m scripts.train_gnn -d $PROCESSED_TRAINING_DATA_DIR -o $LOG_OUTPUT_DIR -r "${MODEL_NAME}_gnn" -m GSpool -k 1
echo "Finished step 1"

gnn_weight_file="${LOG_OUTPUT_DIR}/${MODEL_NAME}_gnn_f1.pt"
echo "Starting step 2 of 4: Generating GNN Predictions as Logits"
python -m scripts.generate_gnn_predictions -d $PROCESSED_TRAINING_DATA_DIR -o $GNN_LOGIT_DIR -w $gnn_weight_file -f logits
echo "Finished step 2"

echo "Starting step 3 of 4: Training CNN"
python -m scripts.train_refinement_cnn -d $PROCESSED_TRAINING_DATA_DIR  -o $LOG_OUTPUT_DIR -l $GNN_LOGIT_DIR -r "${MODEL_NAME}_cnn" -k 1
echo "Finished step 3"

echo "Starting step 4 of 4: Generating Final Predictions"
cnn_weight_file="${LOG_OUTPUT_DIR}/${MODEL_NAME}_cnn_f1.pt"
python -m scripts.generate_joint_predictions -d $PROCESSED_TEST_DATA_DIR -o $PRED_OUTPUT_DIR -c $cnn_weight_file -g $gnn_weight_file -m GSpool
echo "Finished Step 4. Predictions Generated"