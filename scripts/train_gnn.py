
import argparse
from numpy import r_, around
from torch.utils.data import Subset
import os


from utils.training_helpers import *
from utils.hyperparam_helpers import populate_hardcoded_hyperparameters, generate_random_hyperparameters
from data_processing.data_loader import ImageGraphDataset
from model.gnn_model import GNN

import Filepaths

'''
Expects a path to a prepared data directory (created by preprocess script)
Can specify either k fold validation or train on full dataset.
Checkpoints model in the specified log folder.
'''

def train_on_full_dataset(args,hyperparams,progress_file_fd,dataset):
    print("Training on full dataset")
    model = GNN(args.model_type,hyperparams,dataset)
    train_on_fold(model,args.output_dir+os.sep,hyperparams.n_epochs,args.run_name,1)
    train_set_metrics = model.evaluate(Subset(dataset,range(0,len(dataset))))
    document_metrics(progress_file_fd,f"{args.run_name}_full",train_set_metrics)


#if k==1 then train on full dataset
def run_k_fold_val(args,hyperparams,progress_file_fd,dataset,k):
    assert(k>1)
    folds = chunk_dataset_into_folds(dataset,k)
    for k in range(len(folds)):
        s,e = folds[k]
        val_dataset = Subset(dataset,range(s,e))
        ran = list(r_[0:s,e:len(dataset)])
        train_dataset = Subset(dataset,ran)
        print(f"Fold contains {len(train_dataset)} examples")
        model = GNN(args.model_type,hyperparams,train_dataset)
        k=k+1
        train_on_fold(model,args.output_dir+os.sep,hyperparams.n_epochs,args.run_name,k)
        train_set_metrics = model.evaluate(train_dataset)
        document_metrics(progress_file_fd,f"{args.run_name}_f{k}_train",train_set_metrics)
        val_set_metrics = model.evaluate(val_dataset)
        document_metrics(progress_file_fd,f"{args.run_name}_f{k}_val",val_set_metrics)


def document_metrics(fp, description, results):
    counts = results[1]
    metrics = results[0]
    metrics = around(metrics,4)
    print(f"\n#{description} Results#")
    print("Loss:",metrics[0])
    print(f"Predicted Node Counts:", counts[0:4])
    print(f"Label Node Counts:", counts[4:8])
    print(f"WT Node Dice: {metrics[1]}, CT Node Dice: {metrics[2]}, ET Node Dice: {metrics[3]}")
    print(f"WT Voxel Dice: {metrics[4]}, CT Voxel Dice: {metrics[5]}, ET Voxel Dice: {metrics[6]}")
    print(f"WT HD95: {metrics[7]}, CT HD95: {metrics[8]}, ET HD95: {metrics[9]}")
    update_progress_file(fp,description,metrics[0],metrics[4:7])




if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default=Filepaths.PROCESSED_DATA_DIR, help='path to the directory where data is stored',type=str)
    parser.add_argument('-o', '--output_dir', default=Filepaths.LOG_DIR,help='Log directory',type=str)
    parser.add_argument('-r', '--run_name', default=None,help='A unique name to save results under',type=str)
    parser.add_argument('-m', '--model_type', default="GSpool",help='What graph learning layer to use. GSpool, GSmean, GSgcn, GAT',type=str)
    parser.add_argument('-k', '--num_folds', default=5, help='How many folds to run k fold validation on. 1== train on full dataset',type=int)
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.',type=str)
    parser.add_argument('-x', '--random_hyperparams', default=False,help='whether to generate random hyperparameters',action='store_true')

    args = parser.parse_args()
    dataset = ImageGraphDataset(os.path.expanduser(args.data_dir),args.data_prefix,read_image=False,read_graph=True,read_label=True)

    hyperparams = generate_random_hyperparameters(args.model_type) if args.random_hyperparams else populate_hardcoded_hyperparameters(args.model_type)
    #output dir is where the model weights and progress file are stored, does not output any predictions
    args.output_dir = os.path.expanduser(args.output_dir)
    progress_file_fd = f"{args.output_dir}{os.sep}{args.run_name}.txt"
    create_run_progress_file(progress_file_fd,args.model_type,hyperparams)
    if(args.num_folds==1):
        train_on_full_dataset(args,hyperparams,progress_file_fd,dataset)
    elif(args.num_folds>1):
        run_k_fold_val(args,hyperparams,progress_file_fd,dataset,args.num_folds)
    else:
        raise ValueError("Number of folds must be a positive integer")



