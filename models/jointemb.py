import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from modules import SeqEncoder, BOWEncoder

class JointEmbeder(nn.Module):
    """
    https://arxiv.org/pdf/1508.01585.pdf
    https://arxiv.org/pdf/1908.10084.pdf
    """
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
               
        self.name_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.api_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.tok_encoder=BOWEncoder(config['n_words'],config['emb_size'],config['n_hidden'])
        self.desc_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        #self.fuse1=nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden'])
        #self.fuse2 = nn.Sequential(
        #    nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden']),
        #    nn.BatchNorm1d(config['n_hidden'], eps=1e-05, momentum=0.1),
        #    nn.ReLU(),
        #    nn.Linear(config['n_hidden'], config['n_hidden']),
        #)
        self.w_name = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.w_api = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        self.w_desc = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.fuse3 = nn.Linear(config['n_hidden'], config['n_hidden'])
        self.temperature = 0.07
        self.init_weights()
        
    def init_weights(self):# Initialize Linear Weight 
        for m in [self.w_name, self.w_api, self.w_tok, self.fuse3]:        
            m.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.) 
            
    def code_encoding(self, name, name_len, api, api_len, tokens, tok_len):
        name_repr=self.name_encoder(name, name_len)
        api_repr=self.api_encoder(api, api_len)
        tok_repr=self.tok_encoder(tokens, tok_len)
        #code_repr= self.fuse2(torch.cat((name_repr, api_repr, tok_repr),1))
        code_repr = self.fuse3(torch.tanh(self.w_name(name_repr)+self.w_api(api_repr)+self.w_tok(tok_repr)))
        return code_repr
        
    def desc_encoding(self, desc, desc_len):
        desc_repr=self.desc_encoder(desc, desc_len)
        desc_repr=self.w_desc(desc_repr)
        return desc_repr
    
    def similarity(self, code_vec, desc_vec):
        """
        https://arxiv.org/pdf/1508.01585.pdf 
        """
        assert self.conf['sim_measure'] in ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"
        if self.conf['sim_measure']=='cos':
            return F.cosine_similarity(code_vec, desc_vec)
        elif self.conf['sim_measure']=='poly':
            return (0.5*torch.matmul(code_vec, desc_vec.t()).diag()+1)**2
        elif self.conf['sim_measure']=='sigmoid':
            return torch.tanh(torch.matmul(code_vec, desc_vec.t()).diag()+1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code_vec, desc_vec, 2) # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure']=='euc': return euc_sim                
            sigmoid_sim = torch.sigmoid(torch.matmul(code_vec, desc_vec.t()).diag()+1)
            if self.conf['sim_measure']=='gesd': 
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure']=='aesd':
                return 0.5*(euc_sim+sigmoid_sim)
    
    def forward(self, name, name_len, apiseq, api_len, tokens, tok_len, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len, nl_aug_del, nl_aug_del_len, nl_aug_sub, nl_aug_sub_len, tok_aug_del, tok_aug_del_len, tok_aug_swap, tok_aug_swap_len):
        batch_size=name.size(0)
        code_repr=self.code_encoding(name, name_len, apiseq, api_len, tokens, tok_len)
        desc_anchor_repr=self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr=self.desc_encoding(desc_neg, desc_neg_len)
    
        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr) # [batch_sz x 1]
        
        loss=(self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        tok_aug_del_repr = self.code_encoding(name, name_len, apiseq, api_len, tok_aug_del, tok_aug_del_len)
        tok_aug_swap_repr = self.code_encoding(name, name_len, apiseq, api_len, tok_aug_swap, tok_aug_swap_len)
        tok_contrastive_loss = self.compute_contrastive_loss(tok_aug_del_repr, tok_aug_swap_repr, batch_size)

        nl_aug_del_repr = self.desc_encoding(nl_aug_del, nl_aug_del_len)
        nl_aug_sub_repr = self.desc_encoding(nl_aug_sub, nl_aug_sub_len)
        nl_contrastive_loss = self.compute_contrastive_loss(nl_aug_del_repr, nl_aug_sub_repr, batch_size)
        # print("nl contrastive loss finished")
        return (loss + tok_contrastive_loss + nl_contrastive_loss, loss)

    def compute_contrastive_loss(self, aug1, aug2, batch_size):
        z_i = F.normalize(aug1, dim=0)
        z_j = F.normalize(aug2, dim=0)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().cuda()
        numerator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))
        contrastive_loss = torch.sum(loss_partial) / (2 * batch_size)
        return contrastive_loss
