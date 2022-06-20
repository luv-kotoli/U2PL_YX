import argparse
import copy
import logging
import os
import pprint
import random
import time
from datetime import datetime
from xml import dom
import cv2
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchinfo import summary
#import yaml
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from models.deeplab_v3 import DeepLabV3plus
from utils import set_random_seed, AverageMeter, cal_category_confidence,\
                init_log,load_state,label_onehot,intersectionAndUnion,\
                dice_coef,dice_coef_2
from datasets.builder import get_loader
from datasets.augmentation import generate_unsup_data
from losses.loss_helper import get_criterion,compute_contra_memobank_loss,compute_unsupervised_loss
from lr_helper import get_optimizer,get_scheduler


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
#parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch',default=25,type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--dataset',default='mnms')
parser.add_argument('--cuda',default=True)
parser.add_argument('--num_classes',default=4,type=int,help='4 classes for MNMS dataset and 2 classes for SCGM dataset')
parser.add_argument('--img_size',default=288,type=int)
parser.add_argument('--optimizer',default='AdamW')
parser.add_argument('--lr',default=0.0001,type=float)
parser.add_argument('--use_pretrained',default=True)
parser.add_argument('--lr_mode',default='poly')
parser.add_argument('--use_contra_loss',default=True)
parser.add_argument('--is_eval',default=True)
parser.add_argument('--domain',default='A')

args = parser.parse_args()

def log_images(imgs,tb_logger,epoch,image_name='image',is_masks=False):
    #img_len = len(imgs)
    #rows = img_len // 5 + 1
    figure = plt.figure(figsize=(10,10))

    if is_masks == False:
        for i,batch in enumerate(imgs):
            for j in range(len(batch)):
                plt.subplot(2,5,(i+1)*(j+1))
                plt.imshow(batch[j].detach().cpu().squeeze().permute(1,2,0),cmap='gray')
        
    else :
        for i,batch in enumerate(imgs):
            for j in range(len(batch)):
                plt.subplot(2, 5, (i+1)*(j+1))
                plt.imshow(batch[j].squeeze())
    
    tb_logger.add_figure('Validate {} of epoch:{}'.format(image_name,epoch),figure)
    


def main():
    # define params
    global args, prototype
    seed = args.seed

    device = 'cuda:0' if args.cuda is True else 'cpu'

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    # model save path
    save_path = './checkpoints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = SummaryWriter(
        os.path.join('./log/events_seg/' + current_time)
    )

    # set random seed
    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    # Create network
    model = DeepLabV3plus(class_num=args.num_classes)
    model.cuda()

    module_encoder = [model.Resnet101]
    module_decoder = [model.head]
    

    # define loss criterion and data loader
    #sup_loss_fn =  get_criterion(criterion='ohem',ignore_index=255,thresh=0.7,)
    sup_loss_fn = get_criterion(criterion=None,ignore_index=255)
    #print('main funciton:',args.domain)
    train_loader_sup, train_loader_unsup, val_loader = get_loader(args.dataset,batch_size=args.batch_size,domain=args.domain)

    print('dataloader length: ',len(train_loader_sup),len(train_loader_unsup))
    #assert len(train_loader_sup) == len(train_loader_unsup),\
    #f"labeled data {len(train_loader_sup)} unlabeled data {len(train_loader_unsup)}, imbalance!"

    # Optimizer and lr decay scheduler
    optimizer_type = args.optimizer
    opt_args = {
        'lr':args.lr,
        #'momentum':0.9,
        'weight_decay':0.1
    }
    times = 10 

    params_list = []
    for module in module_encoder:
        params_list.append(
            dict(params=module.parameters(), lr=args.lr)
        )
    for module in module_decoder:
        params_list.append(
            dict(params=module.parameters(), lr=args.lr * times)
        )
    
    optimizer = get_optimizer(params_list, type=optimizer_type,opt_args= opt_args)

    # Teacher model
    model_teacher = DeepLabV3plus(class_num=args.num_classes)
    model_teacher = model_teacher.cuda()

    
    for p in model_teacher.parameters():
        p.requires_grad = False
    
    best_prec = 0
    last_epoch = 0

    if args.use_pretrained==True :
        lastest_model = os.path.join(save_path, 'ckpt_mnms_{}_use_rank.pth'.format(args.domain))
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, best_dice,last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    else:
        print('Do not use pretrained model')
    
    # build learning rate scheduler
    lr_args = {
        'power':0.9
    }

    lr_scheduler = get_scheduler(
        args.epoch,args.lr_mode,lr_args, len(train_loader_sup), optimizer, start_epoch=last_epoch
    )

    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(args.num_classes):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    memobank_args = {
        'num_queries': 256,
    }

    prototype = torch.zeros(
        (
            args.num_classes,
            memobank_args["num_queries"],
            1,
            256,
        )
    ).cuda()

    # Start to train model
    for epoch in range(last_epoch, args.epoch):
        # some train settings
        super_only_epoch = 0
        # Training
        train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            memobank,
            queue_ptrlis,
            queue_size,
            super_only_epoch=super_only_epoch
        )

        # Validation
        if args.is_eval:
            logger.info("start evaluation")

            if epoch < super_only_epoch:
                prec,dice = validate(model, val_loader, epoch, logger,num_classes=args.num_classes,ignore_label=255)
            else:
                prec,dice = validate(model_teacher, val_loader, epoch, logger,num_classes=args.num_classes,ignore_label=255)

            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_miou": best_prec,
                'best_dice':best_dice
            }
            if prec > best_prec:
                best_prec = prec
                best_dice = dice
                torch.save(
                    state, os.path.join(save_path, "ckpt_mnms_best_{}_use_rank.pth".format(args.domain))
                )

            torch.save(state, os.path.join(save_path, "ckpt_mnms_{}_use_rank.pth".format(args.domain)))

            logger.info(
                "\033[31m * Currently, the best val result is: {:.2f}, dice coef is {:.2f}\033[0m".format(
                    best_prec * 100, best_dice * 100
                )
            )
            tb_logger.add_scalar("mIoU val", prec, epoch)
            tb_logger.add_scalar('dice mean val',dice,epoch)

def test():
    global args

    dice_list = []
    dice_score = 0
    vendor_list = ['B','C','D']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    dice_meter = AverageMeter()

    output_path = './output/mnms/vendor{}_use_rank/'.format(args.domain)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = './checkpoints'
    class_num=args.num_classes
    current_time = time.get_clock_info
    lastest_model = os.path.join(save_path, 'ckpt_mnms_best_{}_use_rank.pth'.format(args.domain))
    _, _, test_loader = get_loader(args.dataset,batch_size=1,domain=args.domain)

    cudnn.enabled = True
    cudnn.benchmark = True

    #current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #tb_logger = SummaryWriter(
    #    os.path.join('./log/events_seg/' + current_time)
    #)

    model = DeepLabV3plus(class_num=class_num)
    model.cuda()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    if not os.path.exists(lastest_model):
        "No checkpoint found in '{}'".format(lastest_model)
    else:
        print(f"Resume model from: '{lastest_model}'")
        load_state(lastest_model, model, optimizer=None, key="model_state")

    model.eval()

    #for i,batch in tqdm(enumerate(test_loader)):
    for i,batch in tqdm(enumerate(test_loader)):
        imgs = batch['img']
        masks = batch['mask']
        #masks_origin = batch['mask_origin']

        imgs = imgs.cuda()
        masks = masks.long().cuda()
        #masks_origin  = masks_origin.float().cuda()
        masks_onehot = label_onehot(masks,args.num_classes)
        masks_onehot = masks_onehot.cuda()

        with torch.no_grad():
            outs = model(imgs)
        
        output = outs["pred"]
        output = F.interpolate(
            output, masks.shape[1:], mode="bilinear", align_corners=True
        )

        # calculate dice coefficient
        #output_1 = output.data.max(1)[1]
        #target_1 = masks
        #dice_score += dice_coef(output_1,target_12)

        # calculate dice coef 2
        output_temp = output.data.max(1)[1]
        output_2 = label_onehot(output_temp,4)
        target_2 = masks_onehot

        dice_temp = 0
        for dim in range(1,4):
            dice_temp+= dice_coef_2(output_2[:,dim,:,:],target_2[:,dim,:,:])
        dice_temp = dice_temp/3
        dice_list.append(dice_temp.cpu().numpy())

        #print(dice_temp/3)

        dice_score += dice_temp

        #img_out = output_2[0,1:4,:,:].permute(1,2,0).cpu().numpy()

        output = output_temp.squeeze().cpu().numpy()
        #print(output.shape)
        output[output==1] = 40
        output[output==2] = 80
        output[output==3] = 120
        cv2.imwrite(os.path.join(output_path,'{}.jpg'.format(i)),output)

    #dice_mean = dice_score/len(test_loader)
    dice_mean = np.mean(dice_list)
    dice_std = np.std(dice_list)
    #print('Dice mean: {:.4f}'.format(dice_mean))
    print('Dice mean: {:.4f}, Dice std: {:.4f}'.format(dice_mean,dice_std))

def test_all():
    global args

    dice_list = []
    dice_score = 0
    len_data = 0
    vendor_list = ['B','C','D']
    
    for domain in vendor_list:
        output_path = './output/mnms/vendor{}_use_rank/'.format(domain)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = './checkpoints'
        class_num=args.num_classes
        current_time = time.get_clock_info
        lastest_model = os.path.join(save_path, 'ckpt_mnms_best_{}_use_rank.pth'.format(domain))
        _, _, test_loader = get_loader(args.dataset,batch_size=1,domain=domain)

        cudnn.enabled = True
        cudnn.benchmark = True

        model = DeepLabV3plus(class_num=class_num)
        model.cuda()

        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            load_state(lastest_model, model, optimizer=None, key="model_state")

        model.eval()

        #for i,batch in tqdm(enumerate(test_loader)):
        for i,batch in tqdm(enumerate(test_loader)):
            imgs = batch['img']
            masks = batch['mask']
            #masks_origin = batch['mask_origin']

            imgs = imgs.cuda()
            masks = masks.long().cuda()
            #masks_origin  = masks_origin.float().cuda()
            masks_onehot = label_onehot(masks,args.num_classes)
            masks_onehot = masks_onehot.cuda()

            with torch.no_grad():
                outs = model(imgs)
            
            output = outs["pred"]
            output = F.interpolate(
                output, masks.shape[1:], mode="bilinear", align_corners=True
            )

            # calculate dice coef 2
            output_temp = output.data.max(1)[1]
            output_2 = label_onehot(output_temp,4)
            target_2 = masks_onehot

            dice_temp = 0
            for dim in range(1,4):
                dice_temp+= dice_coef_2(output_2[:,dim,:,:],target_2[:,dim,:,:])
            dice_temp = dice_temp/3
            dice_list.append(dice_temp.cpu().numpy())

            dice_score += dice_temp

    dice_mean = np.mean(dice_list)
    dice_std = np.std(dice_list)
    #print('Dice mean: {:.4f}'.format(dice_mean))
    print('Dice mean: {:.4f}, Dice std: {:.4f}'.format(dice_mean,dice_std))


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    memobank,
    queue_ptrlis,
    queue_size,
    **kwargs
):
    global prototype
    ema_decay_origin = 0.99

    model.train()

    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_u)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_u) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        data_l = loader_l_iter.next()
        image_l = data_l['img']
        label_l = data_l['mask']
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        data_u = loader_u_iter.next()
        image_u = data_u['img']
        image_u = image_u.cuda()

        if epoch < kwargs['super_only_epoch']:
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            uns_loss = 0 * rep.sum()
            con_loss = 0 * rep.sum()
        else:
            if epoch == kwargs['super_only_epoch']:
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(pred_u_teacher,(h,w),mode='bilinear',align_corners=True)
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            # use cutmix as the augmentation
            aug_mode = 'cutmix'
            if np.random.uniform(0, 1) < 0.5 and aug_mode:
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=aug_mode,
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # supervised loss
            sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )
                
                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # unsupervised loss
            # Dynamic Partition Adjustment

            loss_weight = 1 # unsupervised loss weight
            drop_percent = 80 # init drop percent
            percent_unreliable = (100 - drop_percent) * (1 - epoch / args.epoch)
            drop_percent = 100 - percent_unreliable
            uns_loss = (
                    compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                    )
                    * loss_weight
            )

            # contrastive loss using unreliable pseudo labels
            contra_flag = "none"
            if args.use_contra_loss:
                # set contrastive loss params
                low_rank = 3; high_rank = 20
                low_entropy_threshold = 20
                negative_high_entropy = True
                contra_loss_mode = 'contra'
                contra_loss_params = {
                    'negative_high_entropy': True,
                    'low_rank': 1,
                    'high_rank': 3,
                    'current_class_threshold': 0.3,
                    'current_class_negative_threshold': 1,
                    'unsupervised_entropy_ignore': 80,
                    'low_entropy_threshold': 20,
                    'num_negatives': 50,
                    'num_queries': 256,
                    'temperature': 0.5,
                }
                


                contra_flag = "{}:{}".format(
                    low_rank, high_rank
                )
                alpha_t = low_entropy_threshold * (
                    1 - epoch / args.epoch
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if negative_high_entropy:
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, args.num_classes),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, args.num_classes),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )

                    # print(pred_all.size(),label_l.size(),label_l_small.size())

                if contra_loss_mode=='binary':
                    contra_flag += " BCE"
                    con_loss = compute_binary_memobank_loss(
                        rep_all,
                        torch.cat((label_l_small, label_u_small)).long(),
                        low_mask_all,
                        high_mask_all,
                        prob_all_teacher.detach(),
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    if not contra_loss_params.get("anchor_ema", False):
                        _, con_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            kwargs=contra_loss_params
                        )
                    else:
                        prototype, _, con_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                            kwargs=contra_loss_params,
                        )

                con_loss = (
                    con_loss* loss_weight
                )

            else:
                con_loss = 0 * rep_all.sum()
        #print(sup_loss,uns_loss,con_loss)
        loss = sup_loss + uns_loss + con_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= kwargs.get("super_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1 - 1 / (i_iter - len(loader_l) * kwargs['super_only_epoch']+ 1),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )
        sup_losses.update(sup_loss.item())
        uns_losses.update(uns_loss.item())
        con_losses.update(con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0:
            logger.info(
                #"[{}][{}] "
                "Iter [{}/{}]\t"
                #"Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                #"Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    #cfg["dataset"]["n_sup"],
                    #contra_flag,
                    i_iter,
                    args.epoch * len(loader_u),
                    #data_time=data_times,
                    #batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)


def validate(
    model,
    data_loader,
    epoch,
    logger,
    **kwargs
):
    dice_score = 0 
    model.eval()
    #data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        kwargs["num_classes"],
        kwargs["ignore_label"],
    )

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for batch in data_loader:
        images = batch['img']
        labels = batch['mask']
        #images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        labels_onehot = label_onehot(labels,args.num_classes)

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )

        output_temp = output.data.max(1)[1]
        output_2 = label_onehot(output_temp,4)
        target_2 = labels_onehot

        dice_temp = 0
        for i in range(1,4):
            dice_temp+= dice_coef_2(output_2[:,i,:,:],target_2[:,i,:,:])
        dice_temp = dice_temp/3

        #print(dice_temp/3)

        dice_score += dice_temp

        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    dice_mean = dice_score/len(data_loader)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    for i, iou in enumerate(iou_class):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logger.info(" * epoch {} mIoU {:.2f} dice {:.4f}".format(epoch, mIoU * 100, dice_mean*100))

    return mIoU,dice_mean

if __name__=='__main__':
    #main()
    #test()
    test_all()