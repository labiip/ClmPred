import logging
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from utils import save_config_file, accuracy, save_checkpoint

import torch.utils.data as data
import glob
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from Create_dataset import MyDataset
torch.manual_seed(0)
import time


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(self.args.savepath)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train_loss_iamge(self,train_loss_all):
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss_all) + 1), train_loss_all, label='Training Loss')

        # find position of lowest validation loss

        plt.xlabel('epochs')
        plt.ylabel('loss')
        #plt.ylim(0, 0.01)  # consistent scale
        plt.xlim(0, len(train_loss_all) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.writer.log_dir}/train_loss.png")

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)
        loss_all=[]

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        print(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        print(f"Training with gpu: {not self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            starttime = time.time()
            loss_sum_epoch = 0
            for i, (images_v1, images_v2) in enumerate(train_loader):
                images = torch.cat((images_v1,images_v2), dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    loss_sum_epoch+=loss.data

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1
            loss_all.append(loss_sum_epoch.item()/(self.args.epochs/self.args.batch_size))
            endtime = time.time()
            # warmup for the first 10 epochs
            '''10->0 for test'''
            if epoch_counter >= 0:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tCosttime:{endtime-starttime}")
            if epoch_counter % 10 == 0:
                print(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tCosttime:{endtime-starttime}")
            if (epoch_counter)%100 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter+1)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info("Training has finished.")
        print("Training has finished.")
        # save model checkpoints
        self.train_loss_iamge(loss_all)
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
