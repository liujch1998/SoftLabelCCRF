import random
import numpy as np
import torch

def collect_region_spatial_feats (bbox, image_size):
    '''
    > bbox [xmin, ymin, xmax, ymax] float
    > image_size (width, height, depth)
    < region_spatial_feats (d_region_spatial=5)
    '''

    (width, height, _) = image_size
    [xmin, ymin, xmax, ymax] = bbox
    l = xmin / width
    u = ymin / height
    r = xmax / width
    d = ymax / height
    a = (r - l) * (d - u)
    region_spatial_feats = torch.tensor([l, u, r, d, a])
    return region_spatial_feats

def iou (bbox1, bbox2):
    '''
    > bbox1 [xmin, ymin, xmax, ymax] float
    > bbox2 [xmin, ymin, xmax, ymax] float
    < iou float
    '''

    w = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    h = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if w <= 0 or h <= 0:
        return 0.0
    a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = (w * h) / max(0.1, a1 + a2 - w * h)
    return iou

def merge_bboxes (bbox1, bbox2):
    '''
    > bbox1 [xmin, ymin, xmax, ymax] float
    > bbox2 [xmin, ymin, xmax, ymax] float
    < bbox [xmin, ymin, xmax, ymax] float
    '''

    if bbox1 is None:
        return bbox2
    if bbox2 is None:
        return bbox1
    [xmin1, ymin1, xmax1, ymax1] = bbox1
    [xmin2, ymin2, xmax2, ymax2] = bbox2
    xmin = min(xmin1, xmin2)
    ymin = min(ymin1, ymin2)
    xmax = max(xmax1, xmax2)
    ymax = max(ymax1, ymax2)
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def clip_bbox_to_image (bbox, image_size):
    '''
    > bbox [xmin, ymin, xmax, ymax] float
    > image_size (width, height, depth)
    < bbox [xmin, ymin, xmax, ymax] float
    '''

    (width, height, _) = image_size
    [xmin, ymin, xmax, ymax] = bbox
    xmin = max(0, min(width, xmin))
    ymin = max(0, min(height, ymin))
    xmax = max(0, min(width, xmax))
    ymax = max(0, min(height, ymax))
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def normalize_bbox (bbox, image_size):
    '''
    > bbox [xmin, ymin, xmax, ymax] float
    > image_size (width, height, depth)
    < bbox_norm [xmin, ymin, xmax, ymax] float
    '''

    (width, height, _) = image_size
    [xmin, ymin, xmax, ymax] = bbox
    xmin = xmin / width
    ymin = ymin / height
    xmax = xmax / width
    ymax = ymax / height
    bbox_norm = [xmin, ymin, xmax, ymax]
    return bbox_norm

def unnormalize_bbox (bbox_norm, image_size):
    '''
    > bbox_norm [xmin, ymin, xmax, ymax] float
    > image_size (width, height, depth)
    < bbox [xmin, ymin, xmax, ymax] float
    '''

    (width, height, _) = image_size
    [xmin, ymin, xmax, ymax] = bbox_norm
    xmin = xmin * width
    ymin = ymin * height
    xmax = xmax * width
    ymax = ymax * height
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def parameterize_bbox_error (gold, cand, image_size):
    '''
    > gold [xmin, ymin, xmax, ymax] float
    > cand [xmin, ymin, xmax, ymax] float
    > image_size (width, height, depth)
    < param [bx, by, bw, bh] float
    '''

    gold_norm = normalize_bbox(gold, image_size)
    cand_norm = normalize_bbox(cand, image_size)

    [xmin, ymin, xmax, ymax] = gold_norm
    xg = (xmin + xmax) / 2
    yg = (ymin + ymax) / 2
    wg = xmax - xmin
    hg = ymax - ymin
    [xmin, ymin, xmax, ymax] = cand_norm
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    wc = xmax - xmin
    hc = ymax - ymin

    bx = (xg - xc) / max(0.1, wc)
    by = (yg - yc) / max(0.1, hc)
    bw = np.log(max(0.1, wg) / max(0.1, wc))
    bh = np.log(max(0.1, hg) / max(0.1, hc))
    param = [bx, by, bw, bh]
    return param

def deparameterize_bbox_error (param, cand, image_size):
    '''
    > param [bx, by, bw, bh] float
    > cand [xmin, ymin, xmax, ymax] float
    > image_size (width, height, depth)
    < pred [xmin, ymin, xmax, ymax] float
    '''

    cand_norm = normalize_bbox(cand, image_size)
    
    [xmin, ymin, xmax, ymax] = cand_norm
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    wc = xmax - xmin
    hc = ymax - ymin
    [bx, by, bw, bh] = param
    xp = xc + bx * wc
    yp = yc + by * hc
    wp = wc * np.exp(bw)
    hp = hc * np.exp(bh)

    xmin = xp - wp / 2
    ymin = yp - hp / 2
    xmax = xp + wp / 2
    ymax = yp + hp / 2
    pred_norm = [xmin, ymin, xmax, ymax]

    pred = unnormalize_bbox(pred_norm, image_size)
    return pred

