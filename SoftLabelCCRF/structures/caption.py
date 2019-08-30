import re

from structures.mention import Mention

class Caption:

    def __init__ (self, index, ann):
        self.index = index

        # process annotated caption
        self.ann = ann.lower()
        self.raw = re.sub(r'\]', r'', re.sub(r'\[/en#[0-9]*/([^\s]*)\s', r'', self.ann)).lower()
        self.tok = self.raw.split(' ')

        # setup mentions
        self.mentions = []
        groups = re.findall(r'\[/en#([0-9]*)/([^\s]*)\s([^\]]*)\]', self.ann, re.M | re.I)
        for group in groups:
            (id, cats_str, mention_raw) = group
            cat = cats_str.split('/')[0]
            ptr = self.raw.find(mention_raw)
            start_pos = len(self.raw[:ptr].split(' ')) - 1
            end_pos = start_pos + len(mention_raw.split(' '))
            # TODO: get rid of these enforcements
            end_pos = min(end_pos, len(self.tok))  # enforce validity of end_pos
            end_pos = max(end_pos, 1)  # enforce validity of end_pos
            start_pos = min(start_pos, end_pos - 1)  # enforce validity of start_pos
            pos = (self.index, start_pos, end_pos)
            mention = Mention(id, cat, mention_raw, pos)
            self.mentions.append(mention)

