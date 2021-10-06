
from numpy import random
from collections import namedtuple
import time
from torch.utils.data import random_split


ParamSet = namedtuple("ParamSet",'n_epochs in_feats out_classes lr lr_decay w_decay class_weights layer_sizes feature_dropout gat_heads gat_residuals')

#function to save run hyperparameters and prepare file for writing
def create_run_progress_file(fp,model_type,hp: ParamSet):
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
            f.write(f"Heads\t{hp.att_heads}\n")
            f.write(f"Residuals\t{hp.residuals}\n")

        f.write("Epoch\tPartition\tLoss\tWT_Dice\tCT_Dice\tET_Dice\n\n")



def create_random_hyperparameters():
    print("Generating Random Hyperparameters...")
    R = random.RandomState(int(str(time.time())[-3:]))
    graph_set = "set4_Brats21/"
    lr = R.choice([0.0005,0.001])
    lr_decay=0.98
    l2_reg = R.choice([0.0001,0])
    class_weights=[0.1,R.normal(2,0.5),R.normal(1,0.2),1,R.normal(2,0.5)]
    feature_dropout = 0.0
    depth = R.choice([5,6,7])
    layer_sizes = [64,64]
    n_epochs=400
    input_feats=20
    #GAT only
    attention_drop = 0
    att_heads = R.randint(4,size=depth)+3
    #hidden_layer_size = 64 +R.randint(200)
    negative_slope = 0.2
    residuals = R.binomial(1,p=0.3,size=depth)
    residuals = [True if el==1 else False for el in residuals]
    hyperparams = (n_epochs,input_feats,4, lr, lr_decay, l2_reg, class_weights, layer_sizes,feature_dropout, att_heads, residuals)
    return graph_set, hyperparams


def populate_hardcoded_hyperparameters(model_type):
    print("Using hardcoded hyperparameters")
    #for prediction/evaluation only input_feats, output_classes and layer_sizes are relevant
    output_classes=4
    lr = 0.01
    lr_decay=0.98
    l2_reg = 0.0001
    feature_dropout = 0.0
    if(model_type=="CNN"):
        n_epochs = 3
        input_feats=8
        class_weights = [0.1,1,2,2]
        layer_sizes=[16]
    else:
        n_epochs =45
        input_feats=20
        class_weights = [0.1,1,2,2]
        layer_sizes=[64]*6
    #only relevant if model is GAT
    att_heads = [4,4,3,3,4,4]
    residuals=[False,False,True,False,False,True]

    hyperparams = ParamSet(n_epochs,input_feats,output_classes,lr,lr_decay,l2_reg,class_weights,layer_sizes, feature_dropout, att_heads, residuals)
    return hyperparams


def chunk_dataset_into_folds(dataset,k):
    fold_size = len(dataset)//k
    folds = []
    for i in range(k):
        folds.append((i*fold_size,(i+1)*fold_size))
    return folds


def update_progress_file(fp,description,loss,dices):
    with open(fp, "a") as f:
        f.write(f"{description}\t{loss}\t{dices[0]}\t{dices[1]}\t{dices[2]}\n")


def train_on_fold(model,checkpoint_dir,n_epoch,run_name,fold):
    lowest_loss=1000
    for i in range(1,n_epoch+1):
        epoch_loss=model.run_epoch()
        if(i%10==0):
            #TODO: tune convergence parameters
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
