import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.elmo import ElmoWrapper

class RnnInputEmbedder (nn.Module):

    def __init__ (self, args):

        super(RnnInputEmbedder, self).__init__()

        self.elmo = ElmoWrapper(args)
        self.drop = nn.Dropout(p=args.drop_prob)

    def forward (self, tokenid):
        '''
        > tokenid (batch, seqlen, .) int
        < input_emb (batch, seqlen, d_lang)
        < mask (batch, seqlen) bool
        '''

        token_emb, mask = self.elmo(tokenid)
        input_emb = token_emb
        input_emb = self.drop(input_emb)
        return input_emb, mask

class RnnEncoder (nn.Module):

    def __init__ (self, args):

        super(RnnEncoder, self).__init__()

        self.args = args
        self.d_hidden = args.d_lang // 2

        self.input_embedder = RnnInputEmbedder(args)
        self.lstm = nn.LSTM(args.d_lang, self.d_hidden, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(p=args.drop_prob)

    def forward (self, tokenid, span, cat, image=None):
        '''
        > tokenid (batch, seqlen, .) int
        > span (batch, max_n_mentions, 2) int
        > cat (batch, max_n_mentions, n_cats)
        < out (batch, max_n_mentions, d_lang)
        < cap (batch, d_lang)
        '''

        batch = tokenid.size(0)

        x, x_mask = self.input_embedder(tokenid)
        # x (batch, seqlen, d_lang)
        # x_mask (batch, seqlen) bool

        hidden_init = (
            torch.zeros((2, batch, self.d_hidden), device=self.args.device),  # h_0
            torch.zeros((2, batch, self.d_hidden), device=self.args.device))  # c_0
        hidden_all, _ = self.lstm(x, hidden_init)
        # hidden_all (batch, seqlen, d_lang)
        hidden_all = self.drop(hidden_all)

        m_mask = span[:,:,0] != -1
        # m_mask (batch, max_n_mentions) bool
        m = []
        for i in range(batch):
            c = []
            for j in range(self.args.max_n_mentions):
                if m_mask[i,j] == 1:
                    hidden_forw = hidden_all[i, span[i,j,1].item()-1, :self.d_hidden]
                    # hidden_forw (d_hidden)
                    hidden_back = hidden_all[i, span[i,j,0].item(), -self.d_hidden:]
                    # hidden_back (d_hidden)
                    hidden = torch.cat((hidden_forw, hidden_back), dim=0).unsqueeze(0)
                    # hidden (1, d_lang)
                    c.append(hidden)
                else:
                    c.append(torch.zeros((1, self.args.d_lang), device=self.args.device))
            c = torch.cat(tuple(c), dim=0)
            m.append(c.unsqueeze(0))
            hidden_forw = hidden_all[i, -1, :self.d_hidden]
            hidden_back = hidden_all[i, 0, -self.d_hidden:]
            hidden = torch.cat((hidden_forw, hidden_back), dim=0).unsqueeze(0)
        m = torch.cat(tuple(m), dim=0)
        # m (batch, max_n_mentions, d_lang)

        out = m
        return out, hidden_all

