import torch
import torch.utils.data as data
import glob
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import piexif
import imghdr

from torch.autograd import Variable


class MyDataset(data.Dataset):
    def __init__(self, path, Train=True, Len=-1, resize=-1, tags_v1=[],tags_v2=[],description='none',img_type='jpg', remove_exif=False):
        self.description=description
        if resize != -1:
            color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)

            transform1 = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),

                transforms.ToTensor(),
                # transforms.Normalize([0.5], ([0.5]))
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform2 = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                #GaussianBlur(kernel_size=int(0.1 * resize)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], ([0.5]))
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        img_format = '*.%s' % img_type

        if remove_exif:
            for name in glob.glob(os.path.join(path, img_format)):
                try:
                    piexif.remove(name)
                except Exception:
                    continue

        if Len == -1:
            self.dataset_v1 = [transform1(Image.open(name).convert("RGB")) for name in
                            glob.glob(os.path.join(path, img_format)) if imghdr.what(name)]
            self.dataset_v2 = [transform2(Image.open(name).convert("RGB")) for name in
                            glob.glob(os.path.join(path, img_format)) if imghdr.what(name)]
            self.tags_v1 = torch.Tensor(tags_v1)
            self.tags_v2 = torch.Tensor(tags_v2)
        else:
            self.dataset_v1 = [transform1(Image.open(name).convert("RGB")) for name in
                            glob.glob(os.path.join(path, img_format))[:Len] if imghdr.what(name)]
            self.dataset_v2 = [transform2(Image.open(name).convert("RGB")) for name in
                            glob.glob(os.path.join(path, img_format))[:Len] if imghdr.what(name)]
            self.Train = Train
            self.tags_v1 = torch.Tensor(tags_v1)
            self.tags_v2 = torch.Tensor(tags_v2)

    def __len__(self):
        return len(self.dataset_v1)

    def __getitem__(self, idx):
        return (self.dataset_v1[idx],self.dataset_v2[idx])

if __name__ =="__main__":
    print('import done')