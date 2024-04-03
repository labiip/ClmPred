import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

from resnet_simclr import ResNetSimCLR
##Change to debug
from simclr_Kfold_early import SimCLR_Kfold
from Mydataset import MyDataset
import torch.nn as nn
from finetune_modelv2 import ResNetSimCLR_finetune

# from finetune_model_supervised import ResNetSimCLR_finetune

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset,random_split
import numpy as np
from tools import EarlyStopping

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Fine Tune of PyTorch SimCLR ')
parser.add_argument('-data', metavar='DIR', default='./dataset/label/data_label_256_78_20230427.pt',
                    help='path to dataset')

parser.add_argument('-p','--model_pretrain_name', default="./upstreamtrain/checkpoint_0600.pth.tar",
                    help='path to model_pretrain_dict') 

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--threshold',default=0,type=float,
                    help='Tolerance to near-label prediction results')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=1024, type=int,
                    help='feature dimension (default: 1024)')
parser.add_argument('--log-every-n-steps', default=10, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fold', default=5, type=int, help='the nunber of fold.')
parser.add_argument('--patience',default=300,type=int,help="patience for earlystoping")
parser.add_argument('--freeze_backbone', default=False, type=bool, help='freez backbone.')
parser.add_argument('--savepath',default='',help='result of model saved')
parser.add_argument('--dataset_seed',default=42,type=int,help='result of model saved')

def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        
        args.device = torch.device('cpu')
        args.gpu_index = -1
    #dataset = ContrastiveLearningDataset(args.data)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset=torch.load(args.data)
    print(len(train_dataset))
    model_pretrain=torch.load(args.model_pretrain_name)
    #train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    model_featuredim={'resnet18':512,'resnet34':512,'resnet50':2048,'resnet101':2048}
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    
    ##IF you need supervised learning, just annotate the sentence on the next line
    model.load_state_dict(model_pretrain['state_dict'])

    
    ## modify model for fine_tune: modify fc
    model= ResNetSimCLR_finetune(model,model_featuredim[args.arch])
 


    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                             last_epoch=-1)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)

    dataset=torch.load(args.data)
    lenth=len(dataset)
    validation_rate=0.2 #validation:dataset
    train_test_rate=0.6+validation_rate #train:test,train:train+val
    trainlen=int(lenth*train_test_rate)
    testlen=lenth-trainlen
    

    train_dataset,testdataset=torch.utils.data.random_split(dataset,[trainlen,testlen],generator=torch.Generator().manual_seed(args.dataset_seed))
    #spilt val from train
    test_loader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=True)
    print(trainlen,":",testlen)
    

    kfold = KFold(n_splits=args.fold, shuffle=True,random_state=args.dataset_seed)
    
    prediction_1=[]
    labels_1=[]
    prediction_2=[]
    labels_2=[]
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
            
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR_Kfold(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            prediction_test,labels_test,predicition_val,labels_val=simclr.trainAndtest_Kfold(train_loader,val_loader,test_loader,fold)
        prediction_1.extend(prediction_test)
        labels_1.extend(labels_test)
        prediction_2.extend(predicition_val)
        labels_2.extend(labels_val)

        del model

        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        model.load_state_dict(model_pretrain['state_dict'])
        model= ResNetSimCLR_finetune(model,model_featuredim[args.arch])
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                             last_epoch=-1)

    prediction_1=np.array(prediction_1)
    labels_1=np.array(labels_1)
    # prediction_1 = np.concatenate(prediction_1, axis=0)
    # labels_1 = np.concatenate(labels_1, axis=0)
    prediction_1=prediction_1.reshape(labels_1.shape) 
    
    np.savetxt(f"{simclr.writer.log_dir}/label_test.txt",labels_1)
    np.savetxt(f"{simclr.writer.log_dir}/prediction_test.txt",prediction_1)

    prediction_2=np.array(prediction_2)
    labels_2=np.array(labels_2)
    # prediction_1 = np.concatenate(prediction_1, axis=0)
    # labels_1 = np.concatenate(labels_1, axis=0)
    prediction_2=prediction_2.reshape(labels_2.shape) 
    
    np.savetxt(f"{simclr.writer.log_dir}/label_val.txt",labels_2)
    np.savetxt(f"{simclr.writer.log_dir}/prediction_val.txt",prediction_2)

    simclr.inspect_prediction_testorval_v1(prediction_1,labels_1,simclr.index_calculation(prediction_1,labels_1),'test',5,['b','b','b','b','b','g'])
    
    simclr.inspect_prediction_testorval_v1(prediction_2,labels_2,simclr.index_calculation(prediction_2,labels_2),'val',5,['b','b','b','b','b','g'])
if __name__ == "__main__":
    main()