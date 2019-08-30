import torch
import torch.nn as nn
import torch.nn.functional as F

class AffHead (nn.Module):

    def __init__ (self, args):

        super(AffHead, self).__init__()

        self.linear = nn.Linear(args.d_fusion, args.d_aff)

    def forward (self, fusion):
        '''
        > fusion (batch, max_n_mentions, max_n_regions, d_fusion)
        < aff (batch, max_n_mentions, max_n_regions, d_aff=1)
        '''

        aff = self.linear(fusion)
        return aff

class RegHead (nn.Module):

    def __init__ (self, args):

        super(RegHead, self).__init__()

        self.linear = nn.Linear(args.d_fusion, args.d_reg)

    def forward (self, fusion):
        '''
        > fusion (batch, max_n_mentions, max_n_regions, d_fusion)
        < reg (batch, max_n_mentions, max_n_regions, d_reg=4)
        '''

        reg = self.linear(fusion)
        return reg

