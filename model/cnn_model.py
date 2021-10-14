from numpy import mean
import torch
from torch.utils.data import DataLoader

from .networks import CnnRefinementNet
from . import evaluation

from data_processing.data_loader import collate_refinement_net

from numpy import zeros,mean



class RefinementModel:
    def __init__(self,hyperparameters,train_dataset,logit_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.FloatTensor(hyperparameters.class_weights).to(self.device)
        self.net=CnnRefinementNet(hyperparameters.in_feats,hyperparameters.out_classes,hyperparameters.layer_sizes)   
        self.net.to(self.device)
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=hyperparameters.lr,weight_decay=hyperparameters.w_decay)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, hyperparameters.lr_decay, last_epoch=-1, verbose=False)
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0,collate_fn=collate_refinement_net) if train_dataset is not None else None
        self.logit_dataset = logit_dataset

    def run_epoch(self):
        self.net.train()
        losses=[]
        for mri,img,lab in self.train_loader:
            try:
                gnn_out,tumor_crop_idxs = self.logit_dataset.get_one(mri)
            except FileNotFoundError as e:
                continue
            cnn_in = combine_logits_and_image(torch.FloatTensor(gnn_out),img,tumor_crop_idxs).to(self.device)

            lab=lab[tumor_crop_idxs].unsqueeze(0).to(self.device)
            cnn_out = self.net(cnn_in)

            loss = self.loss_fcn(cnn_out, lab)

            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.lr_decay.step()
        return mean(losses)

    def evaluate(self,dataset=None):
        self.net.eval()
        metrics = zeros((len(dataset),7))
        i = 0
        for mri,img,lab in dataset:
            try:
                gnn_out,tumor_crop_idxs = self.logit_dataset.get_one(mri)
            except FileNotFoundError as e:
                continue
            cnn_in = combine_logits_and_image(torch.FloatTensor(gnn_out),torch.FloatTensor(img),tumor_crop_idxs)
            lab=torch.LongTensor(lab)[tumor_crop_idxs].unsqueeze(0).to(self.device)
            with torch.no_grad():
                cnn_out = self.net(cnn_in)
                loss = self.loss_fcn(cnn_out, lab)
            _, predicted_classes = torch.max(cnn_out, dim=1)
            voxel_metrics = evaluation.calculate_brats_metrics(predicted_classes,lab.detach().cpu().numpy())
            metrics[i][0]=loss.item()
            metrics[i][1:]=voxel_metrics
            i+=1
        avg_metrics = mean(metrics,axis=0)
        return avg_metrics


    def save_weights(self,folder,name):
        torch.save(self.net.state_dict(),f"{folder}{name}.pt")

#expects float tensors with channels last. Cant handle batch dimension atm.
def combine_logits_and_image(gnn_out,img,tumor_crop):
    cnn_input = torch.cat([img,gnn_out],dim=-1)
    cnn_input = cnn_input[tumor_crop]
    return cnn_input.movedim(-1,0).unsqueeze(0)
        
