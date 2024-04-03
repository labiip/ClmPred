import logging
import pandas as pd
import copy
import warnings
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from utils import save_config_file, accuracy, save_checkpoint

from resnet_simclr import ResNetSimCLR
from finetune_modelv2 import ResNetSimCLR_finetune

import torch.utils.data as data
import glob
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import piexif
import imghdr
from Mydataset import MyDataset
torch.manual_seed(42)
import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tools import EarlyStopping

class SimCLR_Kfold(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(self.args.savepath)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.MSELoss().to(self.args.device)


    def train_loss_iamge(self,train_loss_all,val_loss_all,fold):
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss_all) + 1), train_loss_all, label='Training Loss')
        plt.plot(range(1,len(val_loss_all)+1),val_loss_all,label='Validtion Loss')
        minposs = val_loss_all.index(min(val_loss_all)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        #plt.ylim(0, 0.01)  # consistent scale
        plt.xlim(0, len(train_loss_all) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.writer.log_dir}/train_val_loss_{fold:02d}.png")
    
    def freeze_backbone(self):

        for param in self.model.backbone.parameters():
            param.requires_grad = False



    def get_prediction(self,data):
        prediction=[]
        labels_true=[]
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                prediction_1=self.model(images)

                prediction.append(prediction_1.detach().cpu().numpy())
                labels_true.append(labels.detach().cpu().numpy())
        prediction = np.concatenate(prediction, axis=0)
        labels_true = np.concatenate(labels_true, axis=0)
        prediction=prediction.reshape(labels_true.shape)
        
        return prediction,labels_true
    def index_calculation(self,prediction,labels_true):

        Mse= np.mean(np.square(prediction-labels_true))
        Pcc= np.corrcoef(prediction, labels_true)[0, 1]
        Mae= np.mean(np.abs(prediction-labels_true))

        return {'MSE':Mse,'PCC':Pcc,'MAE':Mae}

    def inspect_prediction(self,data,fold):

        prediction,labels_true=self.get_prediction(data)

        target=self.index_calculation(prediction=prediction,labels_true=labels_true)

        plt.figure(figsize=(10, 8))
        plt.scatter(labels_true, prediction, s=40, color='none', marker='o', edgecolors='b')
        plt.title(f"{target}")
        x = np.linspace(-9, 9, 1000)
        y = x
        plt.plot(x, y, color='r')
        plt.xlabel('labels')
        plt.ylabel('prediction')
        #plt.ylim(0, 0.01)  # consistent scale
        #plt.xlim(0, len(train_loss_all) + 1)  # consistent scale
        #plt.grid(True)
       # plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.writer.log_dir}/inspect_prediction_train_fold{fold:02d}.png")
    
    def inspect_prediction_testorval_v1(self,prediction,labels_true,target,testorval,fold,color): 

        plt.figure(figsize=(10, 8))
        plt.scatter(labels_true, prediction, s=80, color=color[fold], marker='.', edgecolors=None)
        plt.title(f"{target}")
        x = np.linspace(-9, 9, 1000)
        y = x
        plt.plot(x, y, color='r')
        plt.xlabel('labels')
        plt.ylabel('prediction')
        #plt.ylim(0, 0.01)  # consistent scale
        #plt.xlim(0, len(train_loss_all) + 1)  # consistent scale
        #plt.grid(True)
       # plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.writer.log_dir}/inspect_prediction_{testorval}_fold{fold:02d}.png")

    
    def train(self, train_loader,validtion_loder,fold):
        
        self.model.train()
        checkpoint_name = 'checkpoint_{:04d}_fold{:02d}.pth.tar'.format(self.args.epochs,fold)
        early_stopping=EarlyStopping(patience=self.args.patience,path=os.path.join(self.writer.log_dir, checkpoint_name),verbose=False)
        scaler = GradScaler(enabled=self.args.fp16_precision)
        if self.args.freeze_backbone :
            self.freeze_backbone()
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        loss_all=[]
        validtion_loss_all=[]
        n_iter = 0
        n_iter2 = 0
        logging.info(f"Start fine tune of SimCLR training for {self.args.epochs} epochs and fold{fold:02d}.")
        print(f"Start fine tune of SimCLR training for {self.args.epochs} epochs and fold{fold}.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        print(f"Training with gpu: {not self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            n_unkown=0
            n_unkown2=0
            starttime = time.time()
            loss_sum_epoch = 0
            val_loss_sum_epoch = 0
            threshold=self.args.threshold
            self.model.train()
            for i, (images, labels) in enumerate(train_loader):
            
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    prediction = self.model(images)
                    
                    features=prediction
                    prediction=prediction.squeeze(-1)
                    loss = (prediction - labels) ** 2
                    loss[loss<threshold] = 0
                    loss=torch.mean(loss)
                    ##loss = self.criterion(prediction, labels)
                    loss_sum_epoch+=loss.data
                
               
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                if n_iter % self.args.log_every_n_steps == 0:

                    self.writer.add_scalar('train loss', loss, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1
                n_unkown+=1
            for i, (images, labels) in enumerate(validtion_loder):
                self.model.eval()
                with torch.no_grad():
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    prediction=self.model(images)    
                    prediction=prediction.squeeze(-1)
                    loss = (prediction - labels) ** 2
                    loss[loss<threshold] = 0
                    loss=torch.mean(loss)
                    ##loss = self.criterion(prediction, labels)
                    val_loss_sum_epoch+=loss.data
                    if n_iter % self.args.log_every_n_steps == 0:
                        self.writer.add_scalar('val loss', loss, global_step=n_iter2)
                    
                    n_iter2 += 1
                    n_unkown2+=1
                
            loss_epoch= loss_sum_epoch.item()/(n_unkown)
            val_loss_epoch=val_loss_sum_epoch.item()/(n_unkown2)
            

            loss_all.append(loss_epoch)
            validtion_loss_all.append(val_loss_epoch)
            endtime = time.time()
            # warmup for the first 10 epochs
            '''10->0 for test'''
            if epoch_counter >= 0:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss_epoch}\tCosttime:{endtime-starttime}\tLearningrate:{self.scheduler.get_lr()[0]}")
            
            early_stopping(val_loss_epoch,self.model)

            if early_stopping.saveon:
                logging.debug(f"Validation loss decreased ({early_stopping.val_loss_min:.6f} --> {val_loss_epoch:.6f}).  Saving model ...")
            else:
                logging.debug(f"EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}")

         
            if epoch_counter % 10 == 0:
                print(f"Epoch: {epoch_counter}\tLoss: {loss_epoch}\tCosttime:{endtime-starttime}")
                if early_stopping.saveon:
                    
                    print(f"Validation loss decreased ({early_stopping.val_loss_min:.6f} --> {val_loss_epoch:.6f}).  Saving model ...")            
                else:
                    print(f"EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}")
            if early_stopping.early_stop:
                print("Early stopping")
                break 
        logging.info("Training has finished.")
        print("Training has finished.")
        # save model checkpoints
        self.train_loss_iamge(loss_all,validtion_loss_all,fold)
        
        #drow scatter diagram of prediction and labels
        self.inspect_prediction(train_loader,fold)
        '''
        checkpoint_name = 'checkpoint_{:04d}_fold{:02d}.pth.tar'.format(self.args.epochs,fold)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        '''
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def test(self,test_loader,testorval,fold):
        
        modelpre=torch.load(os.path.join(self.writer.log_dir, 'checkpoint_{:04d}_fold{:02d}.pth.tar'.format(self.args.epochs,fold)))
        self.model.load_state_dict(modelpre['state_dict'])
        self.model.eval()

        color=['r','b','m','b','y']

        prediction,labels_true=self.get_prediction(test_loader)

        target=self.index_calculation(prediction=prediction,labels_true=labels_true)

        self.inspect_prediction_testorval_v1(prediction,labels_true,target,testorval,fold,color)
        return prediction,labels_true

    def save_latentfeature(self,data_loader):

        self.model.eval()

        feature=[]
        labels_true=[]
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                feature1=self.model(images)
                print("feature's shape:",feature1.shape)
                feature.append(feature1.detach().cpu().numpy())
                labels_true.append(labels.detach().cpu().numpy())
            feature=np.array(feature)
            labels_true=np.array(labels_true)
            feature = np.concatenate(feature, axis=0)
            labels_true = np.concatenate(labels_true, axis=0)
            feature_pd=pd.DataFrame(feature)
            label_pd=pd.DataFrame(labels_true, columns=["label"])
            feature_pd=feature_pd.T
            print(feature.shape[0])
            encodern_columns = ["sample_" + str(i) for i in range(feature.shape[0])]
            feature_pd.columns = encodern_columns
            feature_pd.to_csv(f'{self.writer.log_dir}/feature.csv', index=False)
            label_pd.to_csv(f'{self.writer.log_dir}/label.csv', index=False)
            print("feature's shape:",feature.shape)
            print("label's shape:",labels_true.shape)





    def trainAndtest_Kfold(self, train_loader,val_loader,test_loader,fold):


        self.train(train_loader=train_loader,validtion_loder=val_loader,fold=fold)


        prediction_test,labels_true_test=self.test(test_loader=test_loader,testorval="test",fold=fold)
        prediction_val,labels_true_val=self.test(test_loader=val_loader,testorval="val",fold=fold)
        return prediction_test,labels_true_test,prediction_val,labels_true_val