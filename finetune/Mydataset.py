import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class MyDataset(data.Dataset):
    def __init__(self, csv_path, img_dir,
                 transofrom=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                ])):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transofrom
        self.imgdata = []
        self.labels = []
        self._getdata()
    def __len__(self):
        return len(self.imgdata)
    def _getdata(self):
        for index, row in self.df.iterrows():
            img_name=row['Image Name']
            label=row['SE']
            print(img_name,label)
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img=self.transform(img)
            self.imgdata.append(img)
            self.labels.append(label)
        self.labels=torch.Tensor(self.labels)
    def __getitem__(self, index):

        return (self.imgdata[index], self.labels[index])
    

# transform1 = transforms.Compose([
#                     transforms.RandomHorizontalFlip(p=1),
#                     transforms.Resize(256),
#                     transforms.CenterCrop(256),
#                     transforms.ToTensor(),
#                 ])
# transform2 = transforms.Compose([
#                     transforms.RandomVerticalFlip(p=1),
#                     transforms.Resize(256),
#                     transforms.CenterCrop(256),
#                     transforms.ToTensor(),
#                 ])
# transform3 = transforms.Compose([
#                     transforms.RandomRotation((-30, -10), resample=None, expand=False),
#                     transforms.Resize(256),
#                     transforms.CenterCrop(256),
#                     transforms.ToTensor(),
#                 ])


# dataset1=torch.load('/home_lv/da.chuang/Fundus_image_analysis/simclr/SimCLR-master - v2/SimCLR-master/datasets/data_labels_256_1.pt')
# dataset2=torch.load('/home_lv/da.chuang/Fundus_image_analysis/simclr/SimCLR-master - v2/SimCLR-master/datasets/data_labels_256_2_3.pt')
# data_labels=dataset1+dataset2
# len(data_labels)
# torch.save(data_labels,"data_labels_256_all.pt")
# dataloader = torch.utils.data.DataLoader(data_labels, batch_size=20, shuffle=True)

