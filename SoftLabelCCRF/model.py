import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import RnnEncoder
from modules.fusion import LowRankBilinearFusion
from modules.heads import AffHead, RegHead

class Model (nn.Module):

    def __init__ (self, args):

        super(Model, self).__init__()

        self.args = args

        self.encoder = RnnEncoder(args)
        self.fusion = LowRankBilinearFusion(args)
        self.aff_head = AffHead(args)
        self.reg_head = RegHead(args)

        self.loss_func_ind = nn.KLDivLoss(reduction='sum')
        self.loss_func_reg = nn.SmoothL1Loss(reduction='sum')

        self.tran0 = nn.Linear(self.args.d_region, 16)
        self.tran1 = nn.Linear(self.args.d_region, 16)
        if self.args.tran_context == 'none':
            self.tranc = None
        elif self.args.tran_context == 'm':
            self.tranc = nn.Linear(self.args.d_lang, 16)
        elif self.args.tran_context == 'mlr':
            self.tranc = nn.Linear(self.args.d_lang * 3, 16)
        elif self.args.tran_context == 'mlrg':
            self.tranc = nn.Linear(self.args.d_lang * 4, 16)
        self.tran = nn.Linear(16, 1)

    def forward (self, X):

        if not self.args.crf:
            return self.ind_forward(X)
        else:
            return self.crf_forward(X)

    def _forward (self, X):
        '''
        < mask_men (batch, max_n_mentions, 1, 1)
        < mask_reg (batch, max_n_mentions, max_n_regions, 1)
        '''

        (tokenid, span, cat, image, region, n_mentions, n_regions, _aff, _reg) = X
        lang, hidden_all = self.encoder(tokenid, span, cat, image=image)
        fused = self.fusion(lang, region)
        aff = self.aff_head(fused)
        reg = self.reg_head(fused)
        mask_men = (span[:,:,0:1] != -1).float().unsqueeze(2)
        mask_reg = (_aff >= 0.5).float()
        return aff, reg, mask_men, mask_reg, hidden_all

    def reg_loss (self, reg, _reg, mask_reg):

        loss_reg = self.loss_func_reg(reg * mask_reg, _reg * mask_reg) / 4.0
        if mask_reg.sum().item() != 0.0:
            loss_reg /= mask_reg.sum()
        return loss_reg

    def ind_forward (self, X):

        (tokenid, span, cat, image, region, n_mentions, n_regions, _aff, _reg) = X
        if not self.args.kld:
            for b in range(_aff.size(0)):
                for m in range(n_mentions[b]):
                    idx = _aff[b,m,:n_regions[b],0].argmax()
                    _aff[b,m,:n_regions[b],0] = 0.0
                    _aff[b,m,idx,0] = 1.0
        aff, reg, mask_men, mask_reg, hidden_all = self._forward(X)
        loss_ind = self.ind_loss(aff, _aff, mask_men, n_regions)
        loss_reg = self.reg_loss(reg, _reg, mask_reg)
        loss = loss_ind + self.args.gamma * loss_reg
        if self.training:
            return loss
        else:
            opt = self.ind_decode(n_mentions, n_regions, aff)
            return loss, opt, reg

    def ind_loss (self, aff, _aff, mask_men, n_regions):

        batch = len(n_regions)
        aff_acti = aff.clone()
        for b in range(batch):
            aff_acti[b,:,n_regions[b]:,0] = -1e8
        aff_norm_log = F.log_softmax(aff_acti, dim=2)
        _aff_acti = _aff.clone()
        _aff_acti[_aff_acti < 0.5] = 0.0
        _aff_norm = _aff_acti / _aff_acti.sum(dim=2, keepdim=True)
        _aff_norm[_aff_norm != _aff_norm] = 0.0

        loss_ind = self.loss_func_ind(aff_norm_log * mask_men, _aff_norm * mask_men)
        if mask_men.sum().item() != 0.0:
            loss_ind /= mask_men.sum()
        return loss_ind

    def ind_decode (self, n_mentions, n_regions, aff):

        batch = len(n_mentions)
        opt = []
        for b in range(batch):
            opt.append([aff[b,m,:n_regions[b]].argmax(dim=0).item() for m in range(n_mentions[b])])
        return opt

    def crf_forward (self, X):

        (tokenid, span, cat, image, region, n_mentions, n_regions, _aff, _reg) = X
        if not self.args.kld:
            for b in range(_aff.size(0)):
                for m in range(n_mentions[b]):
                    idx = _aff[b,m,:n_regions[b],0].argmax()
                    _aff[b,m,:n_regions[b],0] = 0.0
                    _aff[b,m,idx,0] = 1.0
        aff, reg, mask_men, mask_reg, hidden_all = self._forward(X)
        loss_crf = self.crf_loss(region, n_mentions, n_regions, aff, _aff, span, hidden_all)
        loss_reg = self.reg_loss(reg, _reg, mask_reg)
        loss = loss_crf + self.args.gamma * loss_reg
        if self.training:
            return loss
        else:
            opt = self.crf_decode(region, n_mentions, n_regions, aff, span, hidden_all)
            return loss, opt, reg

    def crf_loss (self, region, n_mentions, n_regions, aff, _aff, span, hidden_all):

        _aff_acti = _aff.clone()
        _aff_acti[_aff_acti < 0.5] = 0.0
        _aff_norm = _aff_acti / _aff_acti.sum(dim=2, keepdim=True)
        _aff_norm[_aff_norm != _aff_norm] = 0.0
        _aff_norm_log = torch.log(_aff_norm)
        _aff_norm_log[_aff_norm_log == float('-inf')] = -1e8

        alphas = []
        scores = []
        for b in range(region.size(0)):
            n_tags = n_regions[b]
            alpha_ = torch.zeros((n_tags,), device=self.args.device)
            score_ = torch.zeros((n_tags,), device=self.args.device)
            m_prev = None
            for m in range(n_mentions[b]):
                if _aff_acti[b,m,:n_regions[b],0].sum() == 0.0:
                    continue
                if m_prev is not None:
                    context = self.crf_tran_context(span, hidden_all, b, m_prev, m)
                    tran = self.crf_tran(region[b,:n_tags], context)
                    alpha_ = alpha_.unsqueeze(0).expand(n_tags, -1)
                    alpha_ = alpha_ + tran
                    alpha_ = torch.logsumexp(alpha_, dim=1)
                    score_ = score_.unsqueeze(0).expand(n_tags, -1)
                    score_ = score_ + tran
                    score_ = score_ * reward.unsqueeze(0).expand(n_tags, -1)
                    score_ = score_.sum(dim=1)
                m_prev = m
                emit = aff[b,m,:n_tags,0]
                alpha_ = alpha_ + emit
                reward = _aff_norm[b,m,:n_tags,0]
                reward_log = _aff_norm_log[b,m,:n_tags,0]
                score_ = score_ + emit - reward_log
            if m_prev is not None:
                alpha = torch.logsumexp(alpha_, dim=0, keepdim=True)
                score_ = score_ * reward
                score = score_.sum(dim=0, keepdim=True)
            else:
                alpha = torch.zeros((1,), device=self.args.device)
                score = torch.zeros((1,), device=self.args.device)
            alphas.append(alpha)
            scores.append(score)
        forward_score = torch.cat(alphas, dim=0)
        gold_score = torch.cat(scores, dim=0)

        loss_crf = (forward_score - gold_score).sum()
        if sum(n_mentions) != 0:
            loss_crf /= sum(n_mentions)
        return loss_crf

    def crf_decode (self, region, n_mentions, n_regions, aff, span, hidden_all):

        if self.args.decode == 'viterbi':
            return self.crf_decode_viterbi(region, n_mentions, n_regions, aff, span, hidden_all)
        elif self.args.decode == 'smoothing':
            return self.crf_decode_smoothing(region, n_mentions, n_regions, aff, span, hidden_all)
        else:
            raise Exception('Unexpected args.decode')

    def crf_decode_viterbi (self, region, n_mentions, n_regions, aff, span, hidden_all):

        batch = region.size(0)
        opt = []
        for b in range(batch):
            n_tags = n_regions[b]
            score_ = torch.zeros((n_tags,), device=self.args.device)
            backptr__ = []
            m_prev = None
            for m in range(n_mentions[b]):
                if m_prev is not None:
                    context = self.crf_tran_context(span, hidden_all, b, m_prev, m)
                    tran = self.crf_tran(region[b,:n_tags], context)
                    score_ = score_.unsqueeze(0).expand(n_tags, -1)
                    score_ = score_ + tran
                    score_, backptr_ = torch.max(score_, dim=1)
                    backptr__.append(backptr_.tolist())
                m_prev = m
                emit = aff[b,m,:n_tags,0]
                score_ = score_ + emit
            score, backptr = torch.max(score_, dim=0, keepdim=True)
            backptr = backptr.item()
            best_path = [backptr]
            for backptr_ in reversed(backptr__):
                backptr = backptr_[backptr]
                best_path = [backptr] + best_path
            opt.append(best_path)
        return opt

    def crf_decode_smoothing (self, region, n_mentions, n_regions, aff, span, hidden_all):

        batch = region.size(0)
        opt = []
        for b in range(batch):
            n_tags = n_regions[b]
            alpha__ = []
            alpha_ = torch.zeros((n_tags,), device=self.args.device)
            m_prev = None
            for m in range(n_mentions[b]):
                if m_prev is not None:
                    context = self.crf_tran_context(span, hidden_all, b, m_prev, m)
                    tran = self.crf_tran(region[b,:n_tags], context)
                    alpha_ = alpha_.unsqueeze(0).expand(n_tags, -1)
                    alpha_ = alpha_ + tran
                    alpha_ = torch.logsumexp(alpha_, dim=1)
                m_prev = m
                emit = aff[b,m,:n_tags,0]
                alpha_ = alpha_ + emit
                alpha__.append(alpha_.clone())
            beta__ = []
            beta_ = torch.zeros((n_tags,), device=self.args.device)
            m_next = None
            for m in reversed(range(n_mentions[b])):
                if m_next is not None:
                    context = self.crf_tran_context(span, hidden_all, b, m, m_next)
                    tran = self.crf_tran(region[b,:n_tags], context)
                    beta_ = beta_.unsqueeze(1).expand(-1, n_tags)
                    beta_ = beta_ + tran
                    beta_ = torch.logsumexp(beta_, dim=0)
                m_next = m
                beta__.append(beta_.clone())
                emit = aff[b,m,:n_tags,0]
                beta_ = beta_ + emit
            best_path = []
            for m in range(n_mentions[b]):
                gamma_ = alpha__[m] + beta__[n_mentions[b]-1-m]
                idx = gamma_[:n_regions[b]].argmax().item()
                best_path.append(idx)
            opt.append(best_path)
        return opt

    def crf_tran (self, region, context):
        '''
        > region (n_tags, d_region)
        > context (d_lang * 0/1/3/4)
        < tran (n_tags, n_tags) 'tran[j,i] is the transition score from tag i to tag j'
        '''

        tran0 = self.tran0(region).unsqueeze(0)
        tran1 = self.tran1(region).unsqueeze(1)
        if self.args.tran_context == 'none':
            hidden = F.relu(tran0 + tran1)
        else:
            tranc = self.tranc(context).unsqueeze(0).unsqueeze(1)
            hidden = F.relu(tran0 + tran1 + tranc)
        tran = self.tran(hidden).squeeze(-1)
        return tran

    def crf_tran_context (self, span, hidden_all, b, m_prev, m_next):

        if self.args.tran_context == 'none':
            context = None
        elif self.args.tran_context == 'm':
            context = torch.cat((
                hidden_all[b,span[b,m_next,0]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_prev,1],-self.encoder.d_hidden:], 
                ), dim=0)
        elif self.args.tran_context == 'mlr':
            context = torch.cat((
                hidden_all[b,span[b,m_next,0]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_prev,1],-self.encoder.d_hidden:], 
                hidden_all[b,span[b,m_prev,1]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_prev,0],-self.encoder.d_hidden:], 
                hidden_all[b,span[b,m_next,1]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_next,0],-self.encoder.d_hidden:], 
                ), dim=0)
        elif self.args.tran_context == 'mlrg':
            context = torch.cat((
                hidden_all[b,span[b,m_next,0]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_prev,1],-self.encoder.d_hidden:], 
                hidden_all[b,span[b,m_prev,1]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_prev,0],-self.encoder.d_hidden:], 
                hidden_all[b,span[b,m_next,1]-1,:self.encoder.d_hidden], 
                hidden_all[b,span[b,m_next,0],-self.encoder.d_hidden:], 
                hidden_all[b,-1,:self.encoder.d_hidden], 
                hidden_all[b,0,-self.encoder.d_hidden:], 
                ), dim=0)
        else:
            raise Exception('Unexpected args.tran_context')
        return context

