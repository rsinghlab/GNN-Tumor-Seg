
'''
A collection of helper functions used while training a model.
'''

#save hyperparameters to a log file at a specified path
def create_run_progress_file(fp,model_type,hp):
    with open(fp, "w") as f:
        f.write(f"----Model Parameters----\n")
        f.write(f"Model\t{model_type}\n")
        f.write(f"Epochs\t{hp.n_epochs}\n")
        f.write(f"Input Features\t{hp.in_feats}\n")
        f.write(f"LR\t{hp.lr}\n")
        f.write(f"L2Reg\t{hp.w_decay}\n")
        f.write(f"LR Decay\t{hp.lr_decay}\n")
        f.write(f"Layer Sizes\t{hp.layer_sizes}\n")

        #GAT only
        if(model_type=="GAT"):
            f.write(f"Heads\t{hp.gat_heads}\n")
            f.write(f"Residuals\t{hp.gat_residuals}\n")

        f.write("Fold\tLoss\tWT_Dice\tCT_Dice\tET_Dice\n\n")


def chunk_dataset_into_folds(dataset,k):
    fold_size = len(dataset)//k
    folds = []
    for i in range(k):
        folds.append((i*fold_size,(i+1)*fold_size))
    return folds


def update_progress_file(fp,description,loss,dices):
    with open(fp, "a") as f:
        f.write(f"{description}\t{loss}\t{dices[0]}\t{dices[1]}\t{dices[2]}\n")

#Pass in a model (which already contains the training data) and run it for a specified number of epochs.
#The model checkpoints its weights every couple of epochs.
def train_on_fold(model,checkpoint_dir,n_epoch,run_name,fold):
    lowest_loss=1000
    for i in range(1,n_epoch+1):
        epoch_loss=model.run_epoch()
        #TODO: revert
        if(i%1==0):
            print(f"____Epoch {i}_____")
            print(epoch_loss)
            if(i>n_epoch/2 and epoch_loss>lowest_loss+0.001):
                print("Fold terminated early due to converged train loss")
                print(f"Ran for {i} epochs")
                return
            if epoch_loss<lowest_loss:
                lowest_loss=epoch_loss
                #checkpoint fold
                description = f"{run_name}_f{fold}"
                model.save_weights(checkpoint_dir,description)
    print(f"Finished fold {fold} for run {run_name}")
