import os, sys
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Model
from utils.data import load_tokens
from utils.vision import iou, clip_bbox_to_image, deparameterize_bbox_error
from utils.feats import collect_feats
from utils.stat import StatLoss, StatResult

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%Y/%m/%d %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed (seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

'''
X
    tokenid (batch, seqlen, .) int
    span (batch, max_n_mentions, 2) int
    cat (batch, max_n_mentions, n_cats) bool
    image (batch, d_image)
    region (batch, max_n_regions, d_region)
    n_mentions [batch * int]
    n_regions [batch * int]
    _aff (batch, max_n_mentions, max_n_regions, d_aff=1)
    _reg (batch, max_n_mentions, max_n_regions, d_reg=4)
'''

def train (args, model, optimizer, tokens_train, tokens_dev):
    model.train()
    stat_loss = StatLoss()
    for it in range(args.iters):
        samples = random.choices(tokens_train, k=args.batch)
        X = collect_feats(args, tokens=samples)
        optimizer.zero_grad()
        loss = model(X)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        stat_loss.insert(loss.item())
        if (it + 1) % args.print_every == 0:
            logger.info('Iter %d / %d\tloss_train = %.4f' % (it+1, args.iters, stat_loss.loss_avg))
            stat_loss = StatLoss()
        if (it + 1) % args.eval_every == 0:
            eval(args, model, tokens_dev, split='dev')
            model.train()
        if (it + 1) % args.save_every == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
            }
            model_path = os.path.join(args.output_dir, 'model.%06d.pth' % (it+1))
            torch.save(checkpoint, model_path)

def eval (args, model, tokens, split):
    model.eval()
    stat_loss = StatLoss()
    stat_result = StatResult()
    for token in tokens:
        X, instance = collect_feats(args, token=token)
        with torch.no_grad():
            loss, opt, reg = model(X)
        stat_loss.insert(loss.item())
        for c, caption in enumerate(instance.captions):
            for m, mention in enumerate(caption.mentions):
                r = opt[c][m]
                if not args.no_box_regression:
                    mention.bbox_pred = clip_bbox_to_image(deparameterize_bbox_error(reg[c,m,r].tolist(), instance.regions[r].bbox, instance.image_size), instance.image_size)
                else:
                    mention.bbox_pred = instance.regions[r].bbox
        stat_result.insert(instance)
        instance.visualize_prediction(args.output_dir, split)
    logger.info('loss_eval = %.4f' % stat_loss.loss_avg)
    stat_result.print(logger)

def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    parser.add_argument('--model_name', default='model.pth', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--n_train', default=29783, type=int)
    parser.add_argument('--n_dev', default=1000, type=int)
    parser.add_argument('--n_test', default=1000, type=int)
    parser.add_argument('--iters', default=50000, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--drop_prob', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', default=10.0, type=float)
    parser.add_argument('--no_box_regression',action='store_true')
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--seed', default=19980430, type=int)
    parser.add_argument('--print_every', default=500, type=int)
    parser.add_argument('--eval_every', default=5000, type=int)
    parser.add_argument('--save_every', default=5000, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--kld', action='store_true')
    parser.add_argument('--crf', action='store_true')
    parser.add_argument('--tran_context', default='none', type=str, help='Transition score context. One of [none, m, mlr, mlrg]')
    parser.add_argument('--decode', default='none', type=str, help='Decode algo. One of [viterbi, smoothing] when --crf')
    args = parser.parse_args()

    args.d_lang = 1024
    args.max_n_mentions = 20
    args.n_cats = 8
    args.d_image = 1024
    args.d_region_visual = 2048
    args.d_region_spatial = 5
    args.d_region = args.d_region_visual + args.d_region_spatial
    args.max_n_regions = 1000
    args.d_rank = 1024
    args.d_fusion = 1024
    args.d_aff = 1
    args.d_reg = 4

    set_seed(args.seed)

    args.device = torch.device('cuda')

    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        if args.do_train:
            os.makedirs(os.path.join(args.output_dir, 'visualize', 'train'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'visualize', 'dev'), exist_ok=True)
        if args.do_test:
            os.makedirs(os.path.join(args.output_dir, 'visualize', 'test'), exist_ok=True)

    if args.do_train:
        tokens_train = load_tokens('train', args.n_train)
        tokens_dev = load_tokens('dev', args.n_dev)
        
        logger.info('Initializing model ...')
        model = Model(args).to(args.device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9,0.98))

        logger.info('Training model ...')
        train(args, model, optimizer, tokens_train, tokens_dev)

    if args.do_test:
        tokens_test = load_tokens('test', args.n_test)

        logger.info('Loading model ...')
        if not args.do_train:
            model = Model(args).to(args.device)
            model_path = os.path.join(args.output_dir, args.model_name)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        logger.info('Testing model ...')
        eval(args, model, tokens_test, split='test')

if __name__ == '__main__':
    main()

