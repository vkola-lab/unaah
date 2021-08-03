import torch
import numpy as np
import time
import math
import cv2

#helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
    
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def single_mask_color_img(img, mask, color=[255, 255, 0], alpha=0.001):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 200, 200]
    alpha: float [0, 1]. 

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    mask = mask.astype(bool)
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return(out)
    
def double_mask_color_img(img, mask1, mask2, confidence=[0,75,73], uncertainty=[0, 96, 255], alpha=0.001):

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    u = mask1 | mask2
    c = mask1 & mask2
    out = img.copy()
    img_layer = img.copy()
    img_layer[u] = uncertainty
    img_layer[c] = confidence
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return(out)
    
def double_mask_color_img_v2(img, mask1, mask2, cm1=[0,75,73], cm2=[0, 96, 255], alpha=0.005):

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    c = mask1 & mask2
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask1] = cm1
    img_layer[mask2] = cm2
    img_layer[c] = [255,255,255]
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return(out)


def dice_coefficient(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    prediction : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    mask : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    """
    im1 = ~(np.asarray(im1).astype(np.bool))
    im2 = ~(np.asarray(im2).astype(np.bool))

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def iou(predictions, labels, threshold=None, average=True, device=torch.device("cpu"), classes=2,
        ignore_index=255, ignore_background=True):

    """ Calculating Intersection over Union score for semantic segmentation. """

    gt = labels.long().unsqueeze(1).to(device)

    # getting mask for valid pixels, then converting "void class" to background
    valid = gt != ignore_index
    gt[gt == ignore_index] = 0

    # converting to onehot image whith class channels
    onehot_gt_tensor = torch.LongTensor(gt.shape[0], classes, gt.shape[-2], gt.shape[-1]).zero_().to(device)
    onehot_gt_tensor.scatter_(1, gt, 1)  # write ones along "channel" dimension
    classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0

    # check if it's only background
    if ignore_background:
        only_bg = (classes_in_image[:, 1:].sum(dim=1) == 0).sum() > 0
        if only_bg:
            raise ValueError('Image only contains background. Since background is set to ' +
                             'ignored, IoU is invalid.')

    if threshold is None:
        # taking the argmax along channels
        pred = torch.argmax(predictions, dim=1).unsqueeze(1)
        pred_tensor = torch.LongTensor(pred.shape[0], classes, pred.shape[-2], pred.shape[-1]).zero_().to(device)
        pred_tensor.scatter_(1, pred, 1)
    else:
        # counting predictions above threshold
        pred_tensor = (predictions > threshold).long()

    onehot_gt_tensor *= valid.long()
    pred_tensor *= valid.long().to(device)

    intersection = (pred_tensor & onehot_gt_tensor).sum([2, 3]).float()
    union = (pred_tensor | onehot_gt_tensor).sum([2, 3]).float()

    iou = intersection / (union + 1e-12)

    start_id = 1 if ignore_background else 0

    if average:
        average_iou = iou[:, start_id:].sum(dim=1) /\
                      (classes_in_image[:, start_id:].sum(dim=1)).float()  # discard background IoU
        iou = average_iou

    return iou.cpu().numpy()


def soft_iou(pred_tensor, labels, average=True, device=torch.device("cpu"), classes=2,
             ignore_index=255, ignore_background=True):

    """ Soft IoU score for semantic segmentation, based on 10.1109/ICCV.2017.372 """

    gt = labels.long().unsqueeze(1).to(device)

    # getting mask for valid pixels, then converting "void class" to background
    valid = gt != ignore_index
    gt[gt == ignore_index] = 0
    valid = valid.float().to(device)

    # converting to onehot image with class channels
    onehot_gt_tensor = torch.LongTensor(gt.shape[0], classes, gt.shape[-2], gt.shape[-1]).zero_().to(device)
    onehot_gt_tensor.scatter_(1, gt, 1)  # write ones along "channel" dimension
    classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
    onehot_gt_tensor = onehot_gt_tensor.float().to(device)

    # check if it's only background
    if ignore_background:
        only_bg = (classes_in_image[:, 1:].sum(dim=1) == 0).sum() > 0
        if only_bg:
            raise ValueError('Image only contains background. Since background is set to ' +
                             'ignored, IoU is invalid.')

    onehot_gt_tensor *= valid
    pred_tensor *= valid

    # intersection = torch.max(torch.tensor(0).float(), torch.min(pred_tensor, onehot_gt_tensor).sum(dim=[2, 3]))
    # union = torch.min(torch.tensor(1).float(), torch.max(pred_tensor, onehot_gt_tensor).sum(dim=[2, 3]))

    intersection = (pred_tensor * onehot_gt_tensor).sum(dim=[2, 3])
    union = (pred_tensor + onehot_gt_tensor).sum(dim=[2, 3]) - intersection

    iou = intersection / (union + 1e-12)

    start_id = 1 if ignore_background else 0

    if average:
        average_iou = iou[:, start_id:].sum(dim=1) /\
                      (classes_in_image[:, start_id:].sum(dim=1)).float()  # discard background IoU
        iou = average_iou

    return iou.cpu().numpy()


def dice_score(input, target, classes, ignore_index=-100):

    """ Functional dice score calculation. """

    target = target.long().unsqueeze(1)

    # getting mask for valid pixels, then converting "void class" to background
    valid = target != ignore_index
    target[target == ignore_index] = 0
    valid = valid.float()

    # converting to onehot image with class channels
    onehot_target = torch.LongTensor(target.shape[0], classes, target.shape[-2], target.shape[-1]).zero_().cuda()
    onehot_target.scatter_(1, target, 1)  # write ones along "channel" dimension
    # classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
    onehot_target = onehot_target.float()

    # keeping the valid pixels only
    onehot_target = onehot_target * valid
    input = input * valid

    dice = 2 * (input * onehot_target).sum([2, 3]) / ((input**2).sum([2, 3]) + (onehot_target**2).sum([2, 3]))
    return dice.mean(dim=1)

