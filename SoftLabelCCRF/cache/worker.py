'''
import os, sys
import pickle

from FasterRcnnVisionModel import FasterRcnnVisionModel
vision_model = FasterRcnnVisionModel()

from structures.instance import Instance

gpu_id = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

tokens = sys.argv[2:]
for token in tokens:
    instance = Instance(token, vision_model)
    with open('cache/' + token, 'wb') as f:
        pickle.dump(instance, f, protocol=2)
'''

import os, sys
import pickle

from structures.instance import Instance

gpu_id = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import base64
import numpy as np
import csv
import zlib
import time
import mmap

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

import torch

ix2feats = {}
with open('feats/feats.tsv.%s' % gpu_id, "r+b") as f:
    reader = csv.DictReader(f, delimiter='\t', fieldnames=FIELDNAMES)
    for item in reader:
        ix = item['image_id']
        num_boxes = int(item['num_boxes'])
        feats = {}
        feats['bboxes'] = np.frombuffer(base64.decodestring(item['boxes']), dtype=np.float32).reshape((num_boxes, -1)).tolist()
        rvf = np.frombuffer(base64.decodestring(item['features']), dtype=np.float32).reshape((num_boxes, -1)).tolist()
        feats['regions_visual_feats'] = [torch.tensor([feat]) for feat in rvf]
        ix2feats[ix] = feats

for token in ix2feats:
    instance = Instance(token, ix2feats[token]['bboxes'], ix2feats[token]['regions_visual_feats'])
    with open('cache/%s' % token, 'wb') as f:
        pickle.dump(instance, f, protocol=2)

