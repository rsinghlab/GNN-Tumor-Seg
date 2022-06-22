import argparse
from numpy import r_,around
from torch.utils.data import Subset
import os


from utils.training_helpers import *
from utils.hyperparam_helpers import populate_hardcoded_hyperparameters, generate_random_hyperparameters
from data_processing.data_loader import ImageGraphDataset, PredLogitDataset
from model.cnn_model import RefinementModel
import Filepaths


'''
Trains a CNN to refine the predictions made by a GNN
Requires GNN prediction logits to have been generated and saved via generate_gnn_predictions
prior to running this script.
Can either run k fold validation or just train on the whole dataset. In the simple case where you use
the hardcoded 2 layer CNN parameters there is no need to run k fold validation.

Currently training the CNN is always preceded by training the GNN and generating its predictions as logits,
which are fed into the CNN. Training both end to end is not implemented because it is very slow.
'''

    

def train_on_full_dataset(args,hyperparams,progress_file_fd,image_dataset,logit_dataset):
    print("Training on full dataset")
    model = RefinementModel(hyperparams,image_dataset,logit_dataset)
    train_on_fold(model,args.output_dir+os.sep,hyperparams.n_epochs,args.run_name,1)
    train_set_metrics = model.evaluate(Subset(image_dataset,range(0,len(image_dataset))))
    document_metrics(progress_file_fd,f"{args.run_name}_full",train_set_metrics)


#if k==1 then train on full dataset
def run_k_fold_val(args,hyperparams,progress_file_fd,image_dataset,logit_dataset,k):
    assert(k>1)
    folds = chunk_dataset_into_folds(image_dataset,k)
    for k in range(len(folds)):
        s,e = folds[k]
        val_dataset = Subset(image_dataset,range(s,e))
        ran = list(r_[0:s,e:len(image_dataset)])
        train_dataset = Subset(image_dataset,ran)
        print(f"Fold contains {len(train_dataset)} examples")
        model = RefinementModel(hyperparams,image_dataset,logit_dataset)
        k=k+1
        train_on_fold(model,args.output_dir+os.sep,hyperparams.n_epochs,args.run_name,k)
        train_set_metrics = model.evaluate(train_dataset)
        document_metrics(progress_file_fd,f"{args.run_name}_f{k}_train",train_set_metrics)
        val_set_metrics = model.evaluate(val_dataset)
        document_metrics(progress_file_fd,f"{args.run_name}_f{k}_val",val_set_metrics)



def document_metrics(fp, description, metrics):
    metrics = around(metrics,4)
    print(f"\n#{description} Results#")
    print("Loss:",metrics[0])
    print(f"WT Voxel Dice: {metrics[1]}, CT Voxel Dice: {metrics[2]}, ET Voxel Dice: {metrics[3]}")
    print(f"WT HD95: {metrics[4]}, CT HD95: {metrics[5]}, ET HD95: {metrics[6]}")
    update_progress_file(fp,description,metrics[0],metrics[1:4])


if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default=None, help='path to the directory where data is stored',type=str)
    parser.add_argument('-l', '--saved_logit_dir', default=None, help='path to the directory where the gnn output logits are stored',type=str)
    parser.add_argument('-o', '--output_dir', default=None,help='Directory to save graphs to',type=str)
    parser.add_argument('-r', '--run_name', required=True,help='A unique name to save results under',type=str)
    parser.add_argument('-k', '--num_folds', default=5, help='How many folds to run k fold validation on. 1== train on full dataset',type=int)
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.',type=str)
    parser.add_argument('-x', '--random_hyperparams', default=False,help='whether to generate random hyperparameters',action='store_true')

    args = parser.parse_args()
    image_dataset = ImageGraphDataset(os.path.expanduser(args.data_dir),args.data_prefix,read_image=True,read_graph=False,read_label=True)
    logit_dataset = PredLogitDataset(os.path.expanduser(args.saved_logit_dir))

    hyperparams = generate_random_hyperparameters("CNN") if args.random_hyperparams else populate_hardcoded_hyperparameters("CNN")
    #output dir is where the model weights and progress file are stored, does not output any predictions
    args.output_dir = os.path.expanduser(args.output_dir)
    progress_file_fd = f"{args.output_dir}{os.sep}{args.run_name}.txt"
    create_run_progress_file(progress_file_fd,"CNN",hyperparams)
    if(args.num_folds==1):
        train_on_full_dataset(args,hyperparams,progress_file_fd,image_dataset,logit_dataset)
    elif(args.num_folds>1):
        run_k_fold_val(args,hyperparams,progress_file_fd,image_dataset,logit_dataset,args.num_folds)
    else:
        raise ValueError("Number of folds must be a positive integer")