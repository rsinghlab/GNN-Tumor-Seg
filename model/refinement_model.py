from statistics import mean
import torch
from torch.utils.data import DataLoader

from .. import Constants
from .dgl_nets import CnnRefinementNet
from shared import model_evaluation



class RefinementModel:
    def __init__(self,hyperparameters,train_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.FloatTensor(hyperparameters[2]).to(self.device)
        self.net=CnnRefinementNet()   
        self.net.to(self.device)
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=hyperparameters[0],weight_decay=hyperparameters[1])
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, Constants.GAMMA, last_epoch=-1, verbose=False)
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0)
        #self.train_loader=train_dataset


    def run_epoch(self):
        self.net.train()
        losses=[]
        for mri,img,gnn_out,lab in self.train_loader:
            if(img is None):
                continue
            img=img.movedim(-1,1).to(self.device)
            gnn_out=gnn_out.movedim(-1,1).to(self.device)

            lab=lab.to(self.device)
            cnn_out = self.net(img,gnn_out)
            loss = self.loss_fcn(cnn_out, lab)

            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.lr_decay.step()
        return mean(losses)


    
    def evaluate(self,dataset=None,save_preds_folder=None):
        self.net.eval()
        losses = []
        preds_and_labels=[]
        #default to training dataset
        dataloader = self.train_loader if dataset==None else dataset

        for mri,img,gnn_out,lab in dataloader:
            if(img is None):
                continue
            img=torch.FloatTensor(img).movedim(-1,0).unsqueeze(0).to(self.device)
            gnn_out=torch.FloatTensor(gnn_out).movedim(-1,0).unsqueeze(0).to(self.device)
            lab=torch.LongTensor(lab).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cnn_out = self.net(img,gnn_out)
                loss = self.loss_fcn(cnn_out, lab)
            losses.append(loss.item())
            _, predicted_classes = torch.max(cnn_out, dim=1)
            preds_and_labels.append((predicted_classes.squeeze(0).detach().cpu().numpy(),lab.squeeze(0).detach().cpu().numpy()))
        
        mean_loss = mean(losses)
        dices,hds = model_evaluation.calculate_brats_metrics(preds_and_labels)

        return mean_loss,dices,hds



    def save_weights(self,folder,name):
        torch.save(self.net.state_dict(),f"{folder}{name}.pt")

        
