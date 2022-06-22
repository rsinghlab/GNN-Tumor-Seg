
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing.data_loader import ImageGraphDataset, minibatch_graphs
from .networks import init_graph_net
from . import evaluation
from data_processing.graph_io import project_nodes_to_img


BATCH_SIZE=6

'''
#Input#
model_type is a string that determines the type of graph learning layers used (GraphSAGE, GAT)
hyperparameters is a named tuple defined in utils/hyperparam helpers
train_dataset is an ImageGraphDataset with read_graph set to True.
'''

class GNN:
    def __init__(self,model_type,hyperparameters,train_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device", self.device)
        class_weights = torch.FloatTensor(hyperparameters.class_weights).to(self.device)
        self.net=init_graph_net(model_type,hyperparameters)      
        self.net.to(self.device)
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=hyperparameters.lr,weight_decay=hyperparameters.w_decay)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, hyperparameters.lr_decay, last_epoch=-1, verbose=False)
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=minibatch_graphs) if train_dataset is not None else None


    def run_epoch(self):
        self.net.train()
        losses=[]
        for batch_mris,batch_graphs,batch_features,batch_labels in self.train_loader:
            batch_graphs = batch_graphs.to(self.device)
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            logits = self.net(batch_graphs,batch_features)
            loss = self.loss_fcn(logits, batch_labels)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.lr_decay.step()
        return np.mean(losses)

    #must be a Subset of an ImageGraphDataset
    def evaluate(self,dataset:ImageGraphDataset):
        assert(dataset.dataset.read_label==True)
        self.net.eval()
        #metrics stores loss,label counts, node dices,voxel dices,voxel hausdorff
        metrics = np.zeros((len(dataset),10))
        counts = np.zeros((len(dataset),8))
        i=0
        for curr_id,curr_graph,curr_feats,curr_labels in dataset:
            curr_graph = curr_graph.to(self.device)
            curr_feats = torch.FloatTensor(curr_feats).to(self.device)
            curr_labels = torch.LongTensor(curr_labels).to(self.device)
            with torch.no_grad():
                logits = self.net(curr_graph,curr_feats)
                loss = self.loss_fcn(logits, curr_labels)
            _, predicted_classes = torch.max(logits, dim=1)
            predicted_classes=predicted_classes.detach().cpu().numpy()
            metrics[i][0]=loss.item()
            ct, res = self.calculate_all_metrics_for_brain(curr_id,dataset,predicted_classes,curr_labels.detach().cpu().numpy())
            metrics[i][1:] = res
            counts[i]=ct
            i+=1
        avg_metrics = np.mean(metrics,axis=0)
        total_counts = np.sum(counts,axis=0)
        return avg_metrics, total_counts

    #Calculates a slew of different metrics that might be interesting such as the number of nodes of each label and voxel and node Dice scores
    def calculate_all_metrics_for_brain(self,mri_id,dataset,node_preds,node_labels):
        label_counts = np.concatenate([evaluation.count_node_labels(node_preds),evaluation.count_node_labels(node_labels)])
        node_dices = evaluation.calculate_node_dices(node_preds,node_labels)
        #read in voxel_labels and supervoxel mapping to compute the image metrics
        #TODO: dataset.dataset is currently required since the dataset is a torch Subset object created when chunking the 
        #initial dataset into folds. There is likely a more elegant solution than this.
        sv_partitioning = dataset.dataset.get_supervoxel_partitioning(mri_id)
        true_voxels = dataset.dataset.get_voxel_labels(mri_id)
        pred_voxels = project_nodes_to_img(sv_partitioning,node_preds)
        voxel_metrics = evaluation.calculate_brats_metrics(pred_voxels,true_voxels)
        return label_counts,np.concatenate([node_dices,voxel_metrics])

    def save_weights(self,folder,name):
        torch.save(self.net.state_dict(),f"{folder}{name}.pt")
