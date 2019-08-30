import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankBilinearFusion (nn.Module):

    def __init__ (self, args):

        super(LowRankBilinearFusion, self).__init__()

        self.u = nn.Linear(args.d_lang, args.d_rank, bias=False)
        self.v = nn.Linear(args.d_region, args.d_rank, bias=False)
        self.p = nn.Linear(args.d_rank, args.d_fusion)
        self.drop = nn.Dropout(p=args.drop_prob)

    def forward (self, u, v):
        '''
        > u (batch, max_n_mentions, d_u)
        > v (batch, max_n_regions, d_v)
        < o (batch, max_n_mentions, max_n_regions, d_o)
        '''

        o = self.p(self.u(u).unsqueeze(2) * self.v(v).unsqueeze(1))
        o = F.relu(self.drop(o))
        return o

