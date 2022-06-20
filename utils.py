import numpy as np
import torch
import os
import random
import torch.nn.functional as F
from PIL import Image
import logging

def set_random_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cal_category_confidence(
    preds_student_sup, preds_student_unsup, gt, preds_teacher_unsup, num_classes
):
    category_confidence = torch.zeros(num_classes).type(torch.float32)
    preds_student_sup = F.softmax(preds_student_sup, dim=1)
    preds_student_unsup = F.softmax(preds_student_unsup, dim=1)
    for ind in range(num_classes):
        cat_mask_sup_gt = gt == ind
        if torch.sum(cat_mask_sup_gt) == 0:
            value = 0
        else:
            conf_map_sup = preds_student_sup[:, ind, :, :]
            value = torch.sum(conf_map_sup * cat_mask_sup_gt) / (
                torch.sum(cat_mask_sup_gt) + 1e-12
            )
        category_confidence[ind] = value

    return category_confidence

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def load_state(path, model, optimizer=None, key="state_dict"):

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    print(
                        "caution: size-mismatch key: {} size: {} -> {}".format(
                            k, v.shape, v_dst.shape
                        )
                    )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_miou"]
            best_dice = checkpoint['best_dice']
            last_iter = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(
                "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                    path, last_iter
                )
            )
            return best_metric,best_dice, last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))

def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

'''
def dice_coef(output,target,ignore_index=255):
    smooth = 1e-5
    iflat = output.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return (2. * intersection + smooth) / (A_sum + B_sum + smooth)
'''

def dice_coef(output,target):
    assert output.shape==target.shape
    #print(output.unique(),target.unique())
    #output = output/255
    #target = target/255
    output[torch.where(target==255)[0]] = 255
    intersection = (output*target).sum()
    union = (output+target).sum()
    smooth = 1e-4
    dice = 2*intersection/(union+smooth)
    print(dice.data)
    return dice

def dice_coef_2(output,target):
    assert output.shape==target.shape
    smooth = 1e-4
    output_flat = output.view(-1)
    target_flat = target.view(-1)

    output_flat[output_flat!=0] = 1
    target_flat[target_flat!=0] = 1

    intersection = (output_flat*target_flat).sum()
    union = output_flat.sum()+target_flat.sum()

    dice = (2*intersection+smooth) /(union+smooth)
    #print(intersection,output_flat.sum(),target_flat.sum(),dice)
    
    return dice


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    #keys = keys.detach().clone().cpu()
    #gathered_list = gather_together(keys)
    #keys = torch.cat(gathered_list, dim=0).cuda()

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size