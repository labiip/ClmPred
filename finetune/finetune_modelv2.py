import torch.nn as nn
import torchvision.models as models

from exceptions import InvalidBackboneError


class ResNetSimCLR_finetune(nn.Module):

    def __init__(self, pretrain_model, backbone_outdim):
        super(ResNetSimCLR_finetune, self).__init__()
        self.backbone = pretrain_model.backbone
        dim_mlp = backbone_outdim
        # self.backbone.fc=nn.Sequential()

        #self.regressor=nn.Sequential(nn.Linear(dim_mlp,dim_mlp),nn.Tanh(),nn.Linear(dim_mlp,1024),nn.Tanh(),
         #                   nn.Linear(1024,256),nn.Tanh(),nn.Linear(256,1))
        prs_fc=self.backbone.fc[0]
        self.backbone.fc=nn.Sequential()
        self.regressor=nn.Sequential(prs_fc,nn.Tanh(),nn.Linear(dim_mlp,1))  #1

        # self.regressor=nn.Sequential(nn.Linear(dim_mlp,dim_mlp),nn.Tanh(),nn.Linear(dim_mlp,1))   #2
        #self.regressor=nn.Sequential(nn.Linear(dim_mlp,1))   #3 
        # add mlp projection head

    def forward(self, x):
        backbone_out=self.backbone(x)
        regressor_out=self.regressor(backbone_out)
        return regressor_out
    