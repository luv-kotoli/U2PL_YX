import os
from tkinter import Label
from tqdm import tqdm
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

Vendor_A_labeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Labeled/vendorA/'
Vendor_A_mask_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/mask/Labeled/vendorA/'
Vendor_A_unlabeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Unlabeled/vendorA/'

Vendor_B_labeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Labeled/vendorB/'
Vendor_B_mask_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/mask/Labeled/vendorB/'
Vendor_B_unlabeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Unlabeled/vendorB/'

Vendor_C_labeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Labeled/vendorC/'
Vendor_C_mask_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/mask/Labeled/vendorC/'
Vendor_C_unlabeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Unlabeled/vendorC/'

Vendor_D_labeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Labeled/vendorD/'
Vendor_D_mask_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/mask/Labeled/vendorD/'
Vendor_D_unlabeled_dir = 'D:/Hongpeng/Xiong/Projects/SCGM/data/Unlabeled/vendorD/'


def npz_loader(path):
    if not os.path.exists(path):
        OSError('Cannot find the file to load')
    np_array = np.load(path,allow_pickle=True)
    return np_array['arr_0']

def walk_path(path):
    fpaths = []
    if not os.path.exists(path):
        OSError('This direction is not exist')
    for root,_,files in os.walk(path):
        for file in files:
            fpaths.append(os.path.join(root,file))
    return fpaths    


def get_split_data(target_vendor='A',image_size=224,batch_size=1):
    random.seed(14)
    #torch.random.manual_seed(14)
    
    if target_vendor=='A':
        labeled_dir = [Vendor_B_labeled_dir,Vendor_C_labeled_dir,Vendor_D_labeled_dir]
        mask_dir = [Vendor_B_mask_dir,Vendor_C_mask_dir,Vendor_D_mask_dir]
        unlabeled_dir = [Vendor_B_unlabeled_dir,Vendor_C_unlabeled_dir,Vendor_D_unlabeled_dir]

        test_dir = [Vendor_A_labeled_dir]
        test_mask_dir = [Vendor_A_mask_dir]

    elif target_vendor == 'B':
        labeled_dir = [Vendor_A_labeled_dir,Vendor_C_labeled_dir,Vendor_D_labeled_dir]
        mask_dir = [Vendor_A_mask_dir,Vendor_C_mask_dir,Vendor_D_mask_dir]
        unlabeled_dir = [Vendor_A_unlabeled_dir,Vendor_C_unlabeled_dir,Vendor_D_unlabeled_dir]

        test_dir = [Vendor_B_labeled_dir]
        test_mask_dir = [Vendor_B_mask_dir]
    
    elif target_vendor == 'C':
        labeled_dir = [Vendor_A_labeled_dir,Vendor_B_labeled_dir,Vendor_D_labeled_dir]
        mask_dir = [Vendor_A_mask_dir,Vendor_B_mask_dir,Vendor_D_mask_dir]
        unlabeled_dir = [Vendor_A_unlabeled_dir,Vendor_B_unlabeled_dir,Vendor_D_unlabeled_dir]

        test_dir = [Vendor_C_labeled_dir]
        test_mask_dir = [Vendor_C_mask_dir]

    elif target_vendor == 'D':
        labeled_dir = [Vendor_A_labeled_dir,Vendor_B_labeled_dir,Vendor_C_labeled_dir]
        mask_dir = [Vendor_A_mask_dir,Vendor_B_mask_dir,Vendor_C_mask_dir]
        unlabeled_dir = [Vendor_A_unlabeled_dir,Vendor_B_unlabeled_dir,Vendor_C_unlabeled_dir]

        test_dir = [Vendor_D_labeled_dir]
        test_mask_dir = [Vendor_D_mask_dir]

    labeled_dataset = SCGM_Dataset(labeled_dir,mask_dir)
    unlabeled_dataset = SCGM_Dataset(unlabeled_dir,mask_dir=None,is_labeled=False)
    test_dataset = SCGM_Dataset(test_dir,test_mask_dir,is_train=False)

    new_labeldata_num = len(unlabeled_dataset) // len(labeled_dataset)
    new_label_dataset = labeled_dataset
    for i in range(new_labeldata_num):
        new_label_dataset = ConcatDataset([new_label_dataset, labeled_dataset])
    labeled_dataset = new_label_dataset

    labeled_dataloader = DataLoader(labeled_dataset,batch_size,shuffle=True,drop_last=True,pin_memory=False)
    unlabeled_dataloader = DataLoader(unlabeled_dataset,batch_size,shuffle=True,drop_last=True,pin_memory=False)
    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False,drop_last=False,pin_memory=False)
    
    print(len(labeled_dataset),len(unlabeled_dataset),len(test_dataset))

    return labeled_dataloader,unlabeled_dataloader,test_dataloader


class SCGM_Dataset(Dataset):
    def __init__(self,img_dir,mask_dir,is_train=True,is_labeled=True,img_size=288):
        super().__init__()

        if (is_train == False and is_labeled == False):
            print('Error:Test mode need mask labels')
            return

        self.img_paths = []
        self.mask_paths = []

        self.is_labeled = is_labeled
        self.is_train = is_train
        self.new_size = img_size

        if is_train and is_labeled:
            k = 0.2
        else:
            k = 1

        # get img paths
        for path in img_dir:
            tmp_paths = sorted(walk_path(path))
            for i in range(int(len(tmp_paths)*k)):
                self.img_paths.append(tmp_paths[i])
        if mask_dir != None:
            for path in mask_dir:
                tmp_paths = sorted(walk_path(path))
                for i in range(int(len(tmp_paths)*k)):
                    self.mask_paths.append(tmp_paths[i])

        print(img_dir)
        print(len(self.img_paths),len(self.mask_paths))
        
    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        img = npz_loader(img_path)

        img = Image.fromarray(img)
        h,w = img.size

        

        if self.is_labeled:
            # load mask
            mask_path = self.mask_paths[index]
            mask = npz_loader(mask_path)
            mask = mask[:,:,1]
            mask = Image.fromarray(mask)

            # train labeled data
            if self.is_train:
                # rotate, random angle between 0 - 90
                angle = random.randint(0, 90)
                img = F.rotate(img, angle, InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, InterpolationMode.NEAREST)
                if h > 110 and w > 110:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.new_size, self.new_size))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)

                img = transform(img)
                mask = transform(mask)

                img = F.to_tensor(np.array(img)) 
                img = img.repeat(3,1,1)
                mask = F.to_tensor(np.array(mask))
                mask[mask>0.1] = 1
                mask[mask<0.1] = 0
                mask = mask.long()
                mask = torch.squeeze(mask)
                
            # test data
            else:
                if h > 110 and w > 110:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.new_size, self.new_size))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                    transform = transforms.Compose(transform_list)
                img = transform(img)
                mask = transform(mask)

                img = F.to_tensor(np.array(img))
                img = img.repeat(3,1,1)
                mask = F.to_tensor(np.array(mask))
                mask[mask>0.1] = 1
                mask[mask<0.1] = 0
                mask = mask.long()
                mask = torch.squeeze(mask)
                #mask = (mask > 0.1).float()

        # train unlabel data
        else:
            # rotate, random angle between 0 - 90
            angle = random.randint(0, 90)
            img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            if h > 110 and w > 110:
                size = (100, 100)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = [transforms.Resize((self.new_size, self.new_size))] + transform_list
                transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                transform = transforms.Compose(transform_list)
            else:
                size = (100, 100)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = transform_list + [transforms.Resize((self.new_size, self.new_size))]
                transform = transforms.Compose(transform_list)

            img = transform(img)

            img = F.to_tensor(np.array(img))
            img = img.repeat(3,1,1)
            mask = torch.tensor([0])

        ouput_dict = dict(
            img = img,
            mask = mask,
            img_path = img_path
        )

        return ouput_dict # pytorch: N,C,H,W
    
    def __len__(self) -> int:
        return len(self.img_paths)


if __name__ == '__main__':
    labeled_loader,unlabeled_loader,test_loader = get_split_data('A',batch_size=1)

    dataiter = iter(labeled_loader)
    output = dataiter.next()
    img = output['img']
    mask = output['mask']
    img_path = output['img_path']

    plt.subplot(1,2,1)
    plt.imshow(img.squeeze().cpu().permute(1,2,0),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask.cpu().squeeze())
    plt.show()


    #print(img.squeeze().cpu().shape)
    #print(mask.max(),mask.min(),mask.dtype)

    print("img shape",img.shape)
    print("mask shape",mask.shape)
    print('image path',img_path)


    dataiter = iter(unlabeled_loader)
    output = dataiter.next()
    img = output['img']
    mask = output['mask']
    img_path = output['img_path']

    
    print("img shape",img.shape)
    print("mask shape",mask.shape)
    print('image path',img_path)

    dataiter = iter(test_loader)
    output = dataiter.next()
    img = output['img']
    mask = output['mask']
    img_path = output['img_path']


    print("img shape",img.shape)
    print("mask shape",mask.shape)
    print('image path',img_path)