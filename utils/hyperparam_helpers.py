
from numpy import random
from collections import namedtuple
from time import time


'''
Hyperparameters are set here. Can either hardcode or generate random ones. Hyperparameters are stored as named tuples.
Training the GNN has different hyperparameters than training the CNN.
'''

FullParamSet = namedtuple("FullParamSet",'n_epochs in_feats out_classes lr lr_decay w_decay class_weights layer_sizes feature_dropout gat_heads gat_residuals')
EvalParamSet = namedtuple("EvalParamSet",'in_feats out_classes layer_sizes gat_heads gat_residuals')

#These I wouldnt change. Feel free to change those hyperparameters which arent assigned a constant value.
DEFAULT_N_CLASSES=4
DEFAULT_LR=0.0001
DEFAULT_LR_DECAY=0.98
DEFAULT_WEIGHT_DECAY=0.0001
DEFAULT_FEATURE_DROPOUT=0

DEFAULT_GNN_IN_FEATS=20
DEFAULT_CNN_IN_FEATS=8

DEFAULT_BACKGROUND_NODE_LOGITS = [[1.0,-1.0,-1.0,-1.0]]


def populate_hardcoded_hyperparameters(model_type):
    print("Using hardcoded hyperparameters")
    if(model_type=="CNN"):
        n_epochs = 1
        input_feats=DEFAULT_CNN_IN_FEATS
        class_weights = [0.1,5,15,15]
        layer_sizes=[16]
    else:
        n_epochs = 10
        input_feats = DEFAULT_GNN_IN_FEATS
        class_weights = [0.1,1,2,2]
        layer_sizes=[256]*4
    #only relevant if model is GAT
    att_heads = [4,4,3,3,4,4]
    residuals=[False,False,True,False,False,True]

    hyperparams = FullParamSet(n_epochs,input_feats,DEFAULT_N_CLASSES,DEFAULT_LR,DEFAULT_LR_DECAY,DEFAULT_WEIGHT_DECAY,class_weights,layer_sizes, DEFAULT_FEATURE_DROPOUT, att_heads, residuals)
    return hyperparams

#for use in hyperparameter search
def generate_random_hyperparameters(model_type):
    print("Generated Random Hyperparameters")
    R = random.RandomState(int(str(time())[-3:]))
    lr = R.choice([0.0001,0.0005,0.001])
    l2_reg = R.choice([0.0001,0])
    feature_dropout = 0.0
    if(model_type=="CNN"):
        #n_epochs = R.choice([50,100,150])
        n_epochs=3
        input_feats=DEFAULT_CNN_IN_FEATS
        class_weights = [0.1,R.normal(5,1),R.normal(10,2),R.normal(10,2)]
        layer_sizes=[16]
    else:
        #n_epochs = R.choice([300,400,500])
        n_epochs=3
        input_feats=DEFAULT_GNN_IN_FEATS
        class_weights = [0.1,R.normal(1,0.2),R.normal(2,0.2),R.normal(2,0.2)]
        layer_sizes=R.choice([3,4,5])*[int(R.choice([64,128,256]))]

    att_heads = R.randint(4,size=len(layer_sizes))+3
    residuals = R.binomial(1,p=0.3,size=len(layer_sizes))
    residuals = [True if el==1 else False for el in residuals]

    hyperparams = FullParamSet(n_epochs,input_feats,DEFAULT_N_CLASSES,lr,DEFAULT_LR_DECAY,l2_reg,class_weights,layer_sizes, feature_dropout, att_heads, residuals)
    return hyperparams
