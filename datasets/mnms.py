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

random.seed(14)


LabeledVendorA_data_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_data/Labeled/vendorA/'
LabeledVendorA_mask_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_mask/Labeled/vendorA/'
ReA_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_re/Labeled/vendorA/'

LabeledVendorB2_data_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_data/Labeled/vendorB/center2/'
LabeledVendorB2_mask_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_mask/Labeled/vendorB/center2/'
ReB2_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_re/Labeled/vendorB/center2/'

LabeledVendorB3_data_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_data/Labeled/vendorB/center3/'
LabeledVendorB3_mask_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_mask/Labeled/vendorB/center3/'
ReB3_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_re/Labeled/vendorB/center3/'

LabeledVendorC_data_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_data/Labeled/vendorC/'
LabeledVendorC_mask_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_mask/Labeled/vendorC/'
ReC_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_re/Labeled/vendorC/'

LabeledVendorD_data_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_data/Labeled/vendorD/'
LabeledVendorD_mask_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_mask/Labeled/vendorD/'
ReD_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_re/Labeled/vendorD/'

UnlabeledVendorC_data_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_data/Unlabeled/vendorC/'
UnReC_dir = 'D:/Hongpeng/Xiong/Projects/MNMS/mnms_split_2D_re/Unlabeled/vendorC/'

Re_dir = [ReA_dir, ReB2_dir, ReB3_dir, ReC_dir, ReD_dir]
Labeled_data_dir = [LabeledVendorA_data_dir, LabeledVendorB2_data_dir, LabeledVendorB3_data_dir, LabeledVendorC_data_dir, LabeledVendorD_data_dir]
Labeled_mask_dir = [LabeledVendorA_mask_dir, LabeledVendorB2_mask_dir, LabeledVendorB3_mask_dir, LabeledVendorC_mask_dir, LabeledVendorD_mask_dir]

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
    torch.manual_seed(14)
    print(target_vendor)

    if target_vendor=='A':
        domain_1_img_dirs = [Labeled_data_dir[1],Labeled_data_dir[2]]
        domain_2_img_dirs = [Labeled_data_dir[3]]
        domain_3_img_dirs = [Labeled_data_dir[4]]

        domain_1_mask_dirs = [Labeled_mask_dir[1],Labeled_mask_dir[2]]
        domain_2_mask_dirs = [Labeled_mask_dir[3]]
        domain_3_mask_dirs = [Labeled_mask_dir[4]]

        domain_1_re_dirs = [Re_dir[1],Re_dir[2]]
        domain_2_re_dirs = [Re_dir[3]]
        domain_3_re_dirs = [Re_dir[4]]

        domain_1_num = [74, 51]
        domain_2_num = [50]
        domain_3_num = [50]

        test_img_dirs = [Labeled_data_dir[0]]
        test_mask_dirs = [Labeled_mask_dir[0]]
        test_re_dirs = [Re_dir[0]]
        test_num = [95]

    elif target_vendor=='B':
        domain_1_img_dirs = [Labeled_data_dir[0]]
        domain_2_img_dirs = [Labeled_data_dir[3]]
        domain_3_img_dirs = [Labeled_data_dir[4]]

        domain_1_mask_dirs = [Labeled_mask_dir[0]]
        domain_2_mask_dirs = [Labeled_mask_dir[3]]
        domain_3_mask_dirs = [Labeled_mask_dir[4]]

        domain_1_re_dirs = [Re_dir[0]]
        domain_2_re_dirs = [Re_dir[3]]
        domain_3_re_dirs = [Re_dir[4]]

        domain_1_num = [95]
        domain_2_num = [50]
        domain_3_num = [50]

        test_img_dirs = [Labeled_data_dir[1],Labeled_data_dir[2]]
        test_mask_dirs = [Labeled_mask_dir[1],Labeled_mask_dir[2]]
        test_re_dirs = [Re_dir[1],Re_dir[2]]
        test_num = [74,51]

    elif target_vendor == 'C':
        domain_1_img_dirs = [Labeled_data_dir[0]]
        domain_2_img_dirs = [Labeled_data_dir[1],Labeled_data_dir[2]]
        domain_3_img_dirs = [Labeled_data_dir[4]]

        domain_1_mask_dirs = [Labeled_mask_dir[0]]
        domain_2_mask_dirs = [Labeled_mask_dir[1],Labeled_mask_dir[2]]
        domain_3_mask_dirs = [Labeled_mask_dir[4]]

        domain_1_re_dirs = [Re_dir[0]]
        domain_2_re_dirs = [Re_dir[1],Re_dir[2]]
        domain_3_re_dirs = [Re_dir[4]]

        domain_1_num = [95]
        domain_2_num = [74,51]
        domain_3_num = [50]

        test_img_dirs = [Labeled_data_dir[3]]
        test_mask_dirs = [Labeled_mask_dir[3]]
        test_re_dirs = [Re_dir[3]]
        test_num = [50]

    elif target_vendor == 'D':
        domain_1_img_dirs = [Labeled_data_dir[0]]
        domain_2_img_dirs = [Labeled_data_dir[1],Labeled_data_dir[2]]
        domain_3_img_dirs = [Labeled_data_dir[3]]

        domain_1_mask_dirs = [Labeled_mask_dir[0]]
        domain_2_mask_dirs = [Labeled_mask_dir[1],Labeled_mask_dir[2]]
        domain_3_mask_dirs = [Labeled_mask_dir[3]]

        domain_1_re_dirs = [Re_dir[0]]
        domain_2_re_dirs = [Re_dir[1],Re_dir[2]]
        domain_3_re_dirs = [Re_dir[3]]

        domain_1_num = [95]
        domain_2_num = [74,51]
        domain_3_num = [50]

        test_img_dirs = [Labeled_data_dir[4]]
        test_mask_dirs = [Labeled_mask_dir[4]]
        test_re_dirs = [Re_dir[4]]
        test_num = [50]

    domain_1_labeled_dataset = MNMS_Dataset(domain_1_img_dirs,domain_1_mask_dirs,domain_1_re_dirs,domain_1_num,img_size=image_size)
    domain_2_labeled_dataset = MNMS_Dataset(domain_2_img_dirs,domain_2_mask_dirs,domain_2_re_dirs,domain_2_num,img_size=image_size)
    domain_3_labeled_dataset = MNMS_Dataset(domain_3_img_dirs,domain_3_mask_dirs,domain_3_re_dirs,domain_3_num,img_size=image_size)
    
    domain_1_unlabeled_dataset = MNMS_Dataset(domain_1_img_dirs,domain_1_mask_dirs,domain_1_re_dirs,domain_1_num,labeled=False,img_size=image_size)
    domain_2_unlabeled_dataset = MNMS_Dataset(domain_2_img_dirs,domain_2_mask_dirs,domain_2_re_dirs,domain_2_num,labeled=False,img_size=image_size)
    domain_3_unlabeled_dataset = MNMS_Dataset(domain_3_img_dirs,domain_3_mask_dirs,domain_3_re_dirs,domain_3_num,labeled=False,img_size=image_size)

    test_dataset = MNMS_Dataset(test_img_dirs,test_mask_dirs,test_re_dirs,test_num,is_train=False,labeled=True,img_size=image_size)
    
    labeled_dataset = ConcatDataset([domain_1_labeled_dataset,domain_2_labeled_dataset,domain_3_labeled_dataset])
    unlabeled_dataset = ConcatDataset([domain_1_unlabeled_dataset,domain_2_unlabeled_dataset,domain_3_unlabeled_dataset])
    
    new_labeldata_num = len(unlabeled_dataset) // len(labeled_dataset) + 1
    new_label_dataset = labeled_dataset
    for i in range(new_labeldata_num):
        new_label_dataset = ConcatDataset([new_label_dataset, labeled_dataset])
    labeled_dataset = new_label_dataset

    labeled_loader = DataLoader(labeled_dataset,batch_size=batch_size,shuffle=True,drop_last=True,pin_memory=False,num_workers=8)
    unlabeled_loader = DataLoader(unlabeled_dataset,batch_size=batch_size,shuffle=True,drop_last=True,pin_memory=False,num_workers=8)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,drop_last=False,pin_memory=False,num_workers=8)

    return  labeled_loader,unlabeled_loader,test_loader



class MNMS_Dataset(Dataset):
    def __init__(self,img_dir,mask_dir,re_dir,domain_num=None,is_train=True,labeled=True,img_size=288):
        super().__init__()

        if (is_train == False and labeled == False):
            print('Error:Test mode need mask labels')
            return

        self.img_paths = []
        self.mask_paths = []
        self.re_paths = []

        tmp_img_paths = []
        tmp_mask_paths = []
        tmp_re_paths = []

        tmp_imgs = []
        tmp_masks = []
        tmp_res = []

        self.new_size= img_size
        self.is_train = is_train
        self.domain_num = domain_num
        self.labeled = labeled

        if is_train:
            k = 0.05
            #k = 0.02

            for num_set in range(len(img_dir)):
                print(img_dir[num_set])
                tmp_img_paths = sorted(walk_path(img_dir[num_set]))
                tmp_mask_paths = sorted(walk_path(mask_dir[num_set]))
                tmp_re_paths = sorted(walk_path(re_dir[num_set]))
                for img_num in range(len(tmp_img_paths)):
                    label_num = math.ceil(domain_num[num_set]*k)+1
                    #print(label_num)
                    if '00{}'.format(label_num) == tmp_img_paths[img_num][-10:-7] or '0{}'.format(label_num)== tmp_img_paths[img_num][-10:-7]:
                        break

                    for mask_num in range(len(tmp_mask_paths)):
                        if tmp_img_paths[img_num][-10:-4] == tmp_mask_paths[mask_num][-10:-4]:
                            tmp_imgs.append(tmp_img_paths[img_num])
                            tmp_masks.append(tmp_mask_paths[mask_num])
                            tmp_res.append(tmp_re_paths[img_num])
                        else:
                            pass
                if labeled != True:
                    self.img_paths.extend([path for path in tmp_img_paths if path not in tmp_imgs])
                    self.re_paths.extend([path for path in tmp_re_paths if path not in tmp_res])
            
            if labeled == True:
                self.img_paths = tmp_imgs
                self.mask_paths = tmp_masks
                self.re_paths = tmp_res
        


        elif not is_train:
            for num_set in range(len(img_dir)):
                print(img_dir[num_set])
                tmp_img_paths = sorted(walk_path(img_dir[num_set]))
                tmp_mask_paths = sorted(walk_path(mask_dir[num_set]))
                tmp_re_paths = sorted(walk_path(re_dir[num_set]))

                for img_num in range(len(tmp_img_paths)):
                    for mask_num in range(len(tmp_mask_paths)):
                        if tmp_img_paths[img_num][-10:-4] == tmp_mask_paths[mask_num][-10:-4]:
                            tmp_imgs.append(tmp_img_paths[img_num])
                            tmp_masks.append(tmp_mask_paths[mask_num])
                            tmp_res.append(tmp_re_paths[img_num])
                        else:
                            pass
            
            self.img_paths = tmp_imgs
            self.mask_paths = tmp_masks
            self.re_paths = tmp_res

        print(len(self.img_paths),'   ',len(self.mask_paths),'   ',len(self.re_paths))

    def __getitem__(self, index: int):
        
        re_path = self.re_paths[index]
        img_path = self.img_paths[index]

        re = npz_loader(re_path)[0]
        img = npz_loader(img_path)
        
        p5 = np.percentile(img.flatten(), 0.5)
        p95 = np.percentile(img.flatten(), 99.5)
        img = np.clip(img, p5, p95) 

        img -= img.min()
        img /= img.max()
        img = img.astype('float32')

        img_tensor = F.to_tensor(np.array(img))
        img_size = img_tensor.size()

        crop_size = 300

        if self.labeled == True:
            if self.is_train:
                mask_path = self.mask_paths[index]
                mask = Image.open(mask_path)

                img = Image.fromarray(img)
                
                # rotate, random angle between 0 - 90
                angle = random.randint(0, 90)
                img = F.rotate(img, angle, InterpolationMode.BILINEAR)

                # rotate, random angle between 0 - 90
                mask = F.rotate(mask, angle, InterpolationMode.NEAREST)

                ## Find the region of mask
                norm_mask = F.to_tensor(np.array(mask))
                region = norm_mask[0] + norm_mask[1] + norm_mask[2]
                non_zero_index = torch.nonzero(region == 1, as_tuple=False)
                if region.sum() > 0:
                    len_m = len(non_zero_index[0])
                    x_region = non_zero_index[len_m//2][0]
                    y_region = non_zero_index[len_m//2][1]
                    x_region = int(x_region.item())
                    y_region = int(y_region.item())
                else:
                    x_region = norm_mask.size(-2) // 2
                    y_region = norm_mask.size(-1) // 2

                # resize and center-crop to 280x280
                #resize_order = re / 1.1
                resize_order = re
                resize_size_h = int(img_size[-2] * resize_order)
                resize_size_w = int(img_size[-1] * resize_order)

                left_size = 0
                top_size = 0
                right_size = 0
                bot_size = 0
                if resize_size_h < self.new_size:
                    top_size = (self.new_size - resize_size_h) // 2
                    bot_size = (self.new_size - resize_size_h) - top_size
                if resize_size_w < self.new_size:
                    left_size = (self.new_size - resize_size_w) // 2
                    right_size = (self.new_size - resize_size_w) - left_size

                transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))]
                transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
                transform = transforms.Compose(transform_list)

                img = transform(img)
                
                ## Define the crop index
                if top_size >= 0:
                    top_crop = 0
                else:
                    if x_region > self.new_size//2:
                        if x_region - self.new_size//2 + self.new_size <= norm_mask.size(-2):
                            top_crop = x_region - self.new_size//2
                        else:
                            top_crop = norm_mask.size(-2) - self.new_size
                    else:
                        top_crop = 0

                if left_size >= 0:
                    left_crop = 0
                else:
                    if y_region > self.new_size//2:
                        if y_region - self.new_size//2 + self.new_size <= norm_mask.size(-1):
                            left_crop = y_region - self.new_size//2
                        else:
                            left_crop = norm_mask.size(-1) - self.new_size
                    else:
                        left_crop = 0

                # random crop to 224x224
                img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)
                # random flip
                hflip_p = random.random()
                img = F.hflip(img) if hflip_p >= 0.5 else img
                vflip_p = random.random()
                img = F.vflip(img) if vflip_p >= 0.5 else img
                img = F.to_tensor(np.array(img))
                # Gaussian bluring:
                transform_list = [transforms.GaussianBlur(5, sigma=(0.25, 1.25))]
                transform = transforms.Compose(transform_list)
                img = transform(img)
                img  = img.repeat(3,1,1)
                
                transform_mask_list = [transforms.Pad(
                    (left_size, top_size, right_size, bot_size))]
                transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                         interpolation=InterpolationMode.NEAREST)] + transform_mask_list
                transform_mask = transforms.Compose(transform_mask_list)

                mask = transform_mask(mask)  # C,H,W
                # random crop to 224x224
                mask = F.crop(mask, top_crop, left_crop, self.new_size, self.new_size)

                # random flip
                mask = F.hflip(mask) if hflip_p >= 0.5 else mask
                mask = F.vflip(mask) if vflip_p >= 0.5 else mask

                mask = F.to_tensor(np.array(mask))
                mask_origin = mask.type(torch.long)

                mask_0 = mask[0]
                mask_1 = mask[1]
                mask_2 = mask[2]

                mask_0[mask_0==1.0]=1
                mask_1[mask_1==1.0]=2
                mask_2[mask_2==1.0]=3
                sc_mask = torch.tensor((np.zeros((1,self.new_size, self.new_size), dtype=np.int64)))

                mask = sc_mask[0]+mask_0+mask_1+mask_2
                mask = mask.type(torch.long)
                #print(mask.dtype)
                #print(mask.size(), mask.max(), mask.min())
                #mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                #mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                #mask = torch.cat((mask, mask_bg), dim=0)


            else:
                mask_path = self.mask_paths[index]
                mask = Image.open(mask_path)  # numpy, HxWx3
                # resize and center-crop to 280x280

                ## Find the region of mask
                norm_mask = F.to_tensor(np.array(mask))
                region = norm_mask[0] + norm_mask[1] + norm_mask[2]
                non_zero_index = torch.nonzero(region == 1, as_tuple=False)
                if region.sum() > 0:
                    len_m = len(non_zero_index[0])
                    x_region = non_zero_index[len_m//2][0]
                    y_region = non_zero_index[len_m//2][1]
                    x_region = int(x_region.item())
                    y_region = int(y_region.item())
                else:
                    x_region = norm_mask.size(-2) // 2
                    y_region = norm_mask.size(-1) // 2

                resize_order = re / 1.1
                resize_size_h = int(img_size[-2] * resize_order)
                resize_size_w = int(img_size[-1] * resize_order)

                left_size = 0
                top_size = 0
                right_size = 0
                bot_size = 0
                if resize_size_h < self.new_size:
                    top_size = (self.new_size - resize_size_h) // 2
                    bot_size = (self.new_size - resize_size_h) - top_size
                if resize_size_w < self.new_size:
                    left_size = (self.new_size - resize_size_w) // 2
                    right_size = (self.new_size - resize_size_w) - left_size

                # transform_list = [transforms.CenterCrop((crop_size, crop_size))]
                transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))]
                transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
                transform_list = [transforms.ToPILImage()] + transform_list
                transform = transforms.Compose(transform_list)
                img = transform(img)
                img = F.to_tensor(np.array(img))

                ## Define the crop index
                if top_size >= 0:
                    top_crop = 0
                else:
                    if x_region > self.new_size//2:
                        if x_region - self.new_size//2 + self.new_size <= norm_mask.size(-2):
                            top_crop = x_region - self.new_size//2
                        else:
                            top_crop = norm_mask.size(-2) - self.new_size
                    else:
                        top_crop = 0

                if left_size >= 0:
                    left_crop = 0
                else:
                    if y_region > self.new_size//2:
                        if y_region - self.new_size//2 + self.new_size <= norm_mask.size(-1):
                            left_crop = y_region - self.new_size//2
                        else:
                            left_crop = norm_mask.size(-1) - self.new_size
                    else:
                        left_crop = 0

                # random crop to 224x224
                img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)
                img = img.repeat(3, 1, 1)

                # resize and center-crop to 280x280
                # transform_mask_list = [transforms.CenterCrop((crop_size, crop_size))]
                transform_mask_list = [transforms.Pad(
                    (left_size, top_size, right_size, bot_size))]
                transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                         interpolation=InterpolationMode.NEAREST)] + transform_mask_list
                transform_mask = transforms.Compose(transform_mask_list)

                mask = transform_mask(mask)  # C,H,W
                mask = F.crop(mask, top_crop, left_crop, self.new_size, self.new_size)
                mask = F.to_tensor(np.array(mask))
                mask_origin = mask.type(torch.long)

                mask_0 = mask[0]
                mask_1 = mask[1]
                mask_2 = mask[2]

                mask_0[mask_0 == 1.0] = 1
                mask_1[mask_1 == 1.0] = 2
                mask_2[mask_2 == 1.0] = 3
                sc_mask = torch.tensor((np.zeros((1, self.new_size, self.new_size), dtype=np.float32)))

                mask = sc_mask[0] + mask_0 + mask_1 + mask_2
                mask = mask.type(torch.long)

                #print(mask.size(), mask.max(), mask.min())
                #mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                #mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                #mask = torch.cat((mask, mask_bg), dim=0)
        
        # if unlabeled
        else:
            mask = torch.tensor([0])
            mask_origin = torch.tensor([0])
            img = Image.fromarray(img)
            # rotate, random angle between 0 - 90
            angle = random.randint(0, 90)
            img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            # resize and center-crop to 280x280
            resize_order = re / 1.1
            resize_size_h = int(img_size[-2] * resize_order)
            resize_size_w = int(img_size[-1] * resize_order)

            left_size = 0
            top_size = 0
            right_size = 0
            bot_size = 0
            if resize_size_h < crop_size:
                top_size = (crop_size - resize_size_h) // 2
                bot_size = (crop_size - resize_size_h) - top_size
            if resize_size_w < crop_size:
                left_size = (crop_size - resize_size_w) // 2
                right_size = (crop_size - resize_size_w) - left_size

            transform_list = [transforms.CenterCrop((crop_size, crop_size))]
            transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_list
            transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
            transform = transforms.Compose(transform_list)

            img = transform(img)

            # random crop to 224x224
            top_crop = random.randint(0, crop_size - self.new_size)
            left_crop = random.randint(0, crop_size - self.new_size)
            img = F.crop(img, top_crop, left_crop, self.new_size, self.new_size)
            # random flip
            hflip_p = random.random()
            vflip_p = random.random()
            img = F.hflip(img) if hflip_p >= 0.5 else img
            img = F.vflip(img) if vflip_p >= 0.5 else img

            img = F.to_tensor(np.array(img))
            # Gaussian bluring:
            transform_list = [transforms.GaussianBlur(5, sigma=(0.25, 1.25))]
            transform = transforms.Compose(transform_list)
            img = transform(img)
            img = img.repeat(3, 1, 1)

        output_dict = dict(
            img = img,
            mask = mask,
            mask_origin = mask_origin 
        )

        return output_dict

    def __len__(self):
        return len(self.img_paths)
                

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #test_vendor = 'A'
    
    #domain_1_labeled_dataset,domain_2_labeled_dataset,domain_3_labeled_dataset,\
    #domain_1_unlabeled_dataset,domain_2_unlabeled_dataset,domain_3_unlabeled_dataset,\
    labeled_loader,unlabeled_loader,test_loader = get_split_data('A',batch_size=1)


    dataiter = iter(labeled_loader)
    output = dataiter.next()
    img = output['img']
    mask = output['mask']
    img_path = output['img_path']
    plt.subplot(1,2,1)
    plt.imshow(img.cpu().squeeze().permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(mask.cpu().squeeze())
    plt.show()

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
                

