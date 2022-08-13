# GNN-Tumor-Seg
This repository accompanies the publication A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation"(["Offical"](https://doi.org/10.1007/978-3-031-08999-2_30), ["Arxiv"](https://arxiv.org/abs/2109.05580)). It contains the source code for the experiments described in the paper as well as the Docker model submitted to the BraTS2021 competition as a package.


## Requirements
Python >=3.7

Numpy>=1.17

Scipy>=1.4

Scikit-image

Networkx

Nibabel

Pytorch>=1.7

DGL>=0.4

Requests/Synapseclient (for data)


## How to use
What follows is a general overview. Please see the block comments in the individual files for in depth explanations of individual functions.

### Preprocessing
We first preprocess the data provided by the competition into the format used by our model. Specifically, the script preprocess.py accomplishes the following:

1. Normalize and standardize each image of each MRI modality

2. Combine multiple MRI modalitities into one image array

3. Swap labels from BraTS order (0,2,1,4) to more intuitive order (0,1,2,3)

4. Convert image into a graph. This is done by running a supervoxel creation algorithm on the standardized and combined image and converting each supervoxel into a graph node, with edges between neighboring supervoxels.

The converted data is then stored in a separate directory, which is used as input for subsequent scripts, e.g. training a model.

Example: 

    python -m scripts.preprocess_dataset -d ~/project_data/BraTS21_data/raw/train -n 15000 -k 0 -b 0.5 -o ~/project_data/BraTS21_data/processed/train -l _seg.nii.gz -p BraTS2021

The CLI arguments are explained in the script.

We expect input data to be in the format provided by BraTS. For more information see data availability section below.

### Training GNN
Entrypoint: scripts.train_gnn.py

Example: See scripts.train_gnn_randomized_hyperparameters.sh

Using the graphs generated by the preprocessing step, we train a GNN to predict labels for individual graph nodes. Trains a GNN according to the user defined hyperparameters. We support the GraphSAGE and GAT graph layers but different/custom GNN architectures can easily be added.
The neural network itself is agnostic to the fact that input graphs represent MRIs.

Saves trained models and a text file with the results of each fold in the specified output directory.

### Generating GNN Predictions
Entrypoint: scripts.generate_gnn_predictions.py

Example: 

    python -m scripts.generate_gnn_predictions -d ~/project_data/BraTS21_data/processed/train -o ~/project_data/BraTS21_data/logits/train -w ~/code/GNN-Tumor-Seg/logs/savedGNNModel.pt  -f logits

Loads in a saved model and generates predictions on a target dataset. These predictions are always reprojected back to images, i.e. the output is NOT a graph. They can either take the form of GNN output logits (needed for training CNN) or of final predictions, where each value is the predicted class for that voxel. The latter form is also uncropped and reordered back to the original BraTS specifications so performance metrics can be directly calculated. 
When loading in a model, ensure that the hyperparameters in load_net_and_weights match those of the saved model.

While the trained GNN can produce reliable predictions on its own, the identification of the tumor subclasses can be further improved by appending a CNN. The next two steps demonstrate training and predicting with a joint model.

### Training CNN to Refine GNN Predictions
Entrypoint: scripts.train_refinement_cnn.py

Example: See scripts.train_cnn_randomized_hyperparameters.sh

Reads in the logits produced by the previous step and trains a CNN to improve the predictions. The CNN relies on the GNN to identify the gross tumor, and then only refines that particular segment of the predictions. This greatly speeds up CNN memory requirements and therefore training because the input size is much smaller. However, if the GNN misses part of the tumor entirely, the CNN cannot improve that part of the predictions.
See our paper for a complete explanation and accompanying figure.

Saves trained models and a text file with the results of each fold in the specified output directory.

### Generating Joint GNN-CNN Predictions
Entrypoint: scripts.generate_joint_predictions.py

Example: 

    python -m scripts.generate_joint_predictions -d ~/project_data/BraTS21_data/processed/val -o ~/project_data/BraTS21_data/preds/val -c ~/code/GNN-Tumor-Seg/logs/savedCNNModel.pt -g ~/code/GNN-Tumor-Seg/logs/savedGNNModel.pt -m GSgcn

Once both a GNN and CNN have been trained, joint predictions can be generated. Ensure the hyperparameters in load_nets match those of the saved models. The predictions will be saved to the specified output directory. They will conform to the BraTS shape and label order and so can be directly compared.

#### Provided weights
In case you want to skip training and directly generate predictions using a pretrained model, we include the weights of a fully trained GNN and CNN. (The Docker image can also be used for this purpose, see below) The GNN has 7 GraphSAGE pool layers with 256 neurons each followed by a ReLU nonlinearity. The CNN uses the default two layer architecture.


### Visualizing Predictions
There are two visualization scripts. Both allow the user to specify the directory of the raw data, the directory of the predictions, the id/name of the sample to visualize, as well as whether to plot the ground truth segmentation (if available). 

plot_pred_volume plots horizontal slices of the input modalities, the predicted segmentation, and (optionally) the ground truth segmentation. The height of the horizontal slices can be navigated by using the j and k keys while the focus is on the matplotlib window.

Example: 

    python -m visualization.plot_pred_volume -d "~/project_data/BraTS21_data/raw/val" -s "~/project_data/BraTS21_data/preds/val" -i BraTS2021_01203 --plot_gt

plot_pred_slices additionally takes as input a coronal, horizontal, and sagittal slice height. It will then display 3 columns of images, one for each plane. In order to look at different slices, the script must be rerun with different arguments.

Example: 

    python -m visualization.plot_pred_slices -d "~/project_data/BraTS21_data/raw/val" -s "~/project_data/BraTS21_data/preds/val" -i BraTS2021_01203 -cp 70 -sp 110 -hp 125

### How to Modify Hyperparameters
Hyperparameters are set in hyperparam_helpers.py. Hardcoded hyperparameters for a single run can be set by directly modifying the values in the populate_hyperparameter_method.
The distributions for random hyperparameter generation can be modified in the generate_random_hyperparameter method.
For the GNN model, the model architecture is searched for via random hyperparameter search. The CNN architecture is not searched for since the deisgn is to purposefully keep it simple.
All possible hyperparamters are generated, even if the desired model doesn't use them.
The hyperparameters must also be adjusted when loading in a saved model. This can be done in the load_net_and_weights function of generate_gnn_predictions.py and generate_joint_predictions.py
This solution works, but is unfortunately not very programmatic.


## Provided Bash Scripts
The train_gnn_randomized_hyperparameters and train_cnn_randomized_hyperparameters are provided as examples of how to train a GNN and CNN model with random hyperparameters. These are intended to be used for hyperparameter tuning.
The scripts will train one model with random hyperparameters. The scripts must be run multiple times (e.g. in parallel) in order to train multiple models with different hyperparameters.
Note that prior to running either script the data must first be preprocessed (see below) and a GNN must be trained prior to training a CNN.

The run_pipeline script is intended to demonstrate the flow of training a complete model and generating final predictions once good hyperparameters have been identified. Unlike those above, it trains a model using set hyperparameters and on the complete dataset rather than chunking it into folds. Note that several steps receive their inputs from the output directories of previous steps. Each step is described in more detail below.

For all of these bash scripts you will of course have to adjust the filepaths. Please also note that prior to running any of these scripts the data must first be preprocessed (see below) and a GNN must be trained prior to training a CNN.

## Using the Docker Image

1. Pull the image: 

        docker pull ghcr.io/rsinghlab/gnn_seg:cpu_build

2. Check to see if image is now available locally using:

        docker images

3. Run the image: 
    
        docker run -it --rm -v "\<path to input folder\>":"/input" -v "\<path to output folder\>":"/output" -e DGLBACKEND=pytorch ghcr.io/rsinghlab/gnn_seg:cpu_build

The docker image contains the weights of a fully trained model and uses these to make the predictions.

The BraTS challenge, and therefore the Docker image, require a particular input format. Notably, the image is NOT run on the entire dataset. Instead, it is run on each MRI individually. As such, the input directory should conform to the following:

input/

   BraTS2021_xxxxx_flair.nii.gz

   BraTS2021_xxxxx_t1.nii.gz

   BraTS2021_xxxxx_t1ce.nii.gz

   BraTS2021_xxxxx_t2.nii.gz

The output will then be produced as:

output/

   xxxxx.nii.gz

In order to segment multiple MRIs sequentially, we recommend either 1) using the generate_joint_predictions.py script or 2) writing a script to run the "docker run" command as many times as needed while changing the input folder argument.

Furthermore, during initial development, we assumed that all provided images would have the same orientation as the BraTS training data:

[ -1.0,  -0.0,  -0.0,  -0.0],

[ -0.0,  -1.0,  -0.0, 239.0],

[  0.0,   0.0,   1.0,   0.0],

[  0.0,   0.0,   0.0,   1.0],

MRIs with a different orientation will likely fail or be segmented incorrectly.

## Data Access
The BraTS data for Task 1 (Segmentation) is currently hosted on Synapse. To learn more about the BraTS challenge please visit https://www.synapse.org/#!Synapse:syn27046444/wiki/ and to access the training and validation data please visit https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

## Citation
Please cite the following if you use the code or model from this repository:

Saueressig, C., Berkley, A., Munbodh, R., Singh, R. (2022). A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation. In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2021. Lecture Notes in Computer Science, vol 12962. Springer, Cham. https://doi.org/10.1007/978-3-031-08999-2_30

```
@inproceedings{saueressig2022joint,
  title={A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation},
  author={Saueressig, Camillo and Berkley, Adam and Munbodh, Reshma and Singh, Ritambhara},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={356--365},
  year={2022},
  organization={Springer}
}
```
