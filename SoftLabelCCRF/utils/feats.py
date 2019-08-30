import pickle
import random

import numpy as np
import torch
from allennlp.modules.elmo import batch_to_ids

from utils.data import load_instance
from utils.vision import iou, parameterize_bbox_error, deparameterize_bbox_error

def collect_feats (args, tokens=None, token=None):
    '''
    > (train) tokens [batch * str]
    > (eval) token str
    < X
    < (eval) instance Instance
    '''

    caption_ = []
    span___ = []
    cat___ = []
    image__ = []
    region___ = []
    n_mentions = []
    n_regions = []
    _aff____ = []
    _reg____ = []

    instances = []
    captions = []

    if tokens is not None:
        for t in tokens:
            instance = load_instance(t)
            instances.append(instance)
            caption = random.choice(instance.captions)
            captions.append(caption)
    if token is not None:
        instance = load_instance(token)
        instances.append(instance)
        for caption in instance.captions:
            instances.append(instance)
            captions.append(caption)

    for (instance, caption) in zip(instances, captions):
        span__ = []
        cat__ = []
        for mention in caption.mentions:
            span__.append(collect_span_feats(mention))
            cat__.append(collect_cat_feats(mention))
        while len(span__) < args.max_n_mentions:
            span__.append(collect_span_feats())
            cat__.append(collect_cat_feats())
        caption_.append(caption.tok)
        span___.append(span__)
        cat___.append(cat__)
        n_mentions.append(len(caption.mentions))

        region__ = []
        for region in instance.regions:
            region__.append(region.region_)
        while len(region__) < args.max_n_regions:
            region__.append(torch.zeros((1, args.d_region)))
        image__.append(instance.image_)
        region___.append(torch.cat(tuple(region__), dim=0).unsqueeze(0))
        n_regions.append(len(instance.regions))

        _aff___ = []
        _reg___ = []
        for mention in caption.mentions:
            _aff__ = []
            _reg__ = []
            for region in instance.regions:
                _aff__.append([iou(mention.bbox, region.bbox)])
                _reg__.append(parameterize_bbox_error(mention.bbox, region.bbox, instance.image_size))
            while len(_aff__) < args.max_n_regions:
                _aff__.append([0.0])
                _reg__.append([0.0, 0.0, 0.0, 0.0])
            _aff___.append(_aff__)
            _reg___.append(_reg__)
        while len(_aff___) < args.max_n_mentions:
            _aff__ = []
            _reg__ = []
            while len(_aff__) < args.max_n_regions:
                _aff__.append([0.0])
                _reg__.append([0.0, 0.0, 0.0, 0.0])
            _aff___.append(_aff__)
            _reg___.append(_reg__)
        _aff____.append(_aff___)
        _reg____.append(_reg___)

    tokenid___ = batch_to_ids(caption_).to(args.device)
    span___ = torch.tensor(span___, device=args.device)
    cat___ = torch.tensor(cat___, device=args.device)
    region___ = torch.cat(tuple(region___), dim=0).to(args.device)
    _aff____ = torch.tensor(_aff____, device=args.device)
    _reg____ = torch.tensor(_reg____, device=args.device)
    X = (tokenid___, span___, cat___, image__, region___, n_mentions, n_regions, _aff____, _reg____)

    if tokens is not None:
        return X
    else:
        return X, instances[0]

def collect_span_feats (mention=None):
    '''
    < span [2 * float]
    '''

    if mention is None:
        return [-1,-1]
    (caption_ix, start_pos, end_pos) = mention.pos
    span = [start_pos, end_pos]
    return span

def collect_cat_feats (mention=None):
    '''
    < cat [n_cats * float]
    '''

    CATS = ['people','clothing','bodyparts','animals','vehicles','instruments','scene','other']
    if mention is None:
        return [0.0 for c in CATS]
    cat = [1.0 if mention.cat == c else 0.0 for c in CATS]
    return cat

