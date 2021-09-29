
from numpy import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataloader import ImageGraphDataset
from .. import Constants
from .dgl_nets import *
from . import evaluation

BATCH_SIZE=6
GAMMA=0.98

class GNN:
    def __init__(self,model_type,hyperparameters,train_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.FloatTensor(hyperparameters[2]).to(self.device)
        self.net=self.init_model(model_type,hyperparameters)      
        self.net.to(self.device)
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=hyperparameters[0],weight_decay=hyperparameters[1])
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, GAMMA, last_epoch=-1, verbose=False)
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
        collate_fn = loaders.minibatch_graphs
        self.train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate_fn)

        
    def init_model(self,model_type,hp):
        if(model_type=='GSpool'):
            net = GraphSage(in_feats=hp.in_feats,layer_size=hp.layer_sizes,n_classes=hp.n_classes,aggregator_type='pool',dropout=hp.dropout)
        elif(model_type=='GSgcn'):
            net = GraphSage(in_feats=hp.in_feats,layer_size=hp.layer_sizes,n_classes=hp.n_classes,aggregator_type='pool',dropout=hp.dropout)
        elif(model_type=='GSmean'):
            net = GraphSage(in_feats=hp.in_feats,layer_size=hp.layer_sizes,n_classes=hp.n_classes,aggregator_type='pool',dropout=hp.dropout)
        elif(model_type=='GAT'):
            net = GAT(in_feats=hp.in_feats,layer_size=hp.layer_sizes,n_classes=hp.n_classes,
                                    heads=hp.gat_heads,residuals=gp.gat_residuals)
        else:
            raise Exception(f"Unknown model type: {model_type}")
        return net


    def run_epoch(self):
        self.net.train()
        losses=[]
        for batch_mris,batch_graphs,batch_features,batch_labels in self.train_loader:
            batch_graphs.to(self.device)
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            logits = self.net(batch_graphs,batch_features)
            loss = self.loss_fcn(logits, batch_labels)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.lr_decay.step()
        return mean(losses)


    #calculate node and voxel dice
    def evaluate(self,dataset:ImageGraphDataset):
        self.net.eval()
        #metrics stores loss,label counts, node dices,voxel dices,voxel hausdorff
        metrics = np.zeros((len(dataset),11))
        counts = np.zeros(len(dataset),4)
        i=0
        for curr_id,curr_graph,curr_feats,curr_labels in dataset:
            curr_graph.to(self.device)
            curr_feats = torch.FloatTensor(curr_feats).to(self.device)
            curr_labels = torch.LongTensor(curr_labels).to(self.device)
            with torch.no_grad():
                logits = self.net(curr_graph,curr_feats)
                loss = self.loss_fcn(logits, curr_labels)
            _, predicted_classes = torch.max(logits, dim=1).detach().cpu().numpy()
            metrics[i][0]=loss
            brain_metrics = calculate_all_metrics_for_brain(curr_id,dataset,predicted_classes,curr_labels.detach().cpu().numpy())
            metrics[i][1:] = brain_metrics[4:]
            counts[i]=brain_metrics[:4]
            i+=1
        avg_metrics = np.mean(metrics,axis=1)
        total_counts = np.sum(counts,axis=1)
        return avg_metrics


    def calculate_all_metrics_for_brain(mri_id,dataset,node_preds,node_labels)
        #read in voxel_labels and supervoxel mapping to compute the image metrics



    def save_weights(self,folder,name):
        torch.save(self.net.state_dict(),f"{folder}{name}.pt")

