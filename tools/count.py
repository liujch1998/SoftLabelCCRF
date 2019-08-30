# Script that produces fig 2(b)

from collections import defaultdict

from utils.data import load_tokens, load_instance
from utils.vision import iou

n_train = 29783
n_dev = 1000
n_test = 1000

tokens_train = load_tokens('train', n_train)
tokens_dev = load_tokens('dev', n_dev)
tokens_test = load_tokens('test', n_test)
tokens = tokens_test

reg2cnt = defaultdict(int)
reg_sum, reg_cnt = 0, 0
for ix, token in enumerate(tokens):
    instance = load_instance(token)
    for caption in instance.captions:
        for mention in caption.mentions:
            reg_cnt += 1
            reg = len([region for region in instance.regions if iou(region.bbox, mention.bbox) >= 0.5])
            reg2cnt[reg] += 1
            reg_sum += reg
reg_avg = reg_sum / reg_cnt
print(reg_avg)
for i in range(20):
    print('%d %d' % (i, reg2cnt[i]))

men2cnt = defaultdict(int)
men_sum, men_cnt = 0, 0
for ix, token in enumerate(tokens):
    instance = load_instance(token)
    for caption in instance.captions:
        men_cnt += 1
        men2cnt[len(caption.mentions)] += 1
        men_sum += len(caption.mentions)
men_avg = men_sum / men_cnt
print(men_avg)
for i in range(20):
    print('%d %d' % (i, men2cnt[i]))

