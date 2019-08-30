from collections import defaultdict

from utils.vision import iou

class StatLoss:

    def __init__ (self):

        self.n, self.loss_sum, self.loss_avg = 0, 0.0, 0.0

    def insert (self, loss):

        self.n += 1
        self.loss_sum += loss
        self.loss_avg = self.loss_sum / self.n

class StatResult:

    def __init__ (self):

        self.all, self.hit, self.acc = 0, 0, 0.0
        self.all_unique, self.hit_unique, self.acc_unique = 0, 0, 0.0
        self.all_multi, self.hit_multi, self.acc_multi = 0, 0, 0.0
        self.intra_cat, self.cross_cat, self.others = 0, 0, 0
        self.all_cat, self.hit_cat, self.acc_cat = defaultdict(int), defaultdict(int), defaultdict(float)

    def insert (self, instance):

        for caption in instance.captions:
            for mention in caption.mentions:
                score = iou(mention.bbox, mention.bbox_pred)
                self.all += 1
                self.hit += 1 if score >= 0.5 else 0
                if any([m.cat == mention.cat for m in caption.mentions if m.id != mention.id]):
                    self.all_multi += 1
                    self.hit_multi += 1 if score >= 0.5 else 0
                else:
                    self.all_unique += 1
                    self.hit_unique += 1 if score >= 0.5 else 0
                if score < 0.5:
                    if any([m.cat == mention.cat and iou(m.bbox, mention.bbox_pred) >= 0.5 for m in caption.mentions]):
                        self.intra_cat += 1
                    elif any([iou(m.bbox, mention.bbox_pred) >= 0.5 for m in caption.mentions]):
                        self.cross_cat += 1
                    else:
                        self.others += 1
                self.all_cat[mention.cat] += 1
                self.hit_cat[mention.cat] += 1 if score >= 0.5 else 0

    def print (self, logger):

        self.acc = self.hit / self.all
        for cat in self.all_cat:
            self.acc_cat[cat] = self.hit_cat[cat] / self.all_cat[cat]
        self.acc_unique = self.hit_unique / self.all_unique
        self.acc_multi = self.hit_multi / self.all_multi
        logger.info('acc = %.4f = %6d / %6d' % (self.acc, self.hit, self.all))
        logger.info('\tacc_unique = %.4f = %6d / %6d' % (self.acc_unique, self.hit_unique, self.all_unique))
        logger.info('\tacc_multi = %.4f = %6d / %6d' % (self.acc_multi, self.hit_multi, self.all_multi))
        logger.info('intra_cat = %d, cross_cat = %d, others = %d' % (self.intra_cat, self.cross_cat, self.others))
        logger.info('Accuracy by entity category:')
        for cat in self.all_cat:
            logger.info('\t%11s %.4f = %6d / %6d' % (cat, self.acc_cat[cat], self.hit_cat[cat], self.all_cat[cat]))
        logger.info('')

