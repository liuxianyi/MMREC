# coding: utf-8
# 
"""
DualGNN: Dual Graph Neural Network for Multimedia Recommendation, IEEE Transactions on Multimedia 2021.
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from common.init import xavier_uniform_initialization


class MMCR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMCR, self).__init__(config, dataset)
    
        dim_x = config['embedding_size']
        has_id = True

        self.i_e_agg = 'sum'
        self.tau = config['tau'] 
        self.m = config['m']
        self.aggr_mode = config['aggr_mode']
        self.num_layer = config['n_layers']
        self.dataset = dataset
        self.reg_weight = config['reg_weight']
        self.drop_rate = config['drop_rate']
        self.dim_latent = config['dim_latent']
        self.cl_weight = config['cl_weight'] / 6.0
        self.hyperNum = config['hyperNum']
        self.hyper_keep_rate = config['hyper_keep_rate']
        self.hyper_layers = config['hyper_layers']

        self.MLP_v = nn.Linear(128, self.dim_latent)
        nn.init.xavier_uniform_(self.MLP_v.weight)
        self.MLP_a = nn.Linear(128, self.dim_latent)
        nn.init.xavier_uniform_(self.MLP_a.weight)
        self.MLP_t = nn.Linear(64, self.dim_latent)
        nn.init.xavier_uniform_(self.MLP_t.weight)

        

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']), allow_pickle=True).item()
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']), allow_pickle=True).item()


        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

    
        self.u_e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.dim_latent), gain=1))
        self.i_e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.dim_latent), gain=1))
        self.u_Hyper = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_latent, self.hyperNum), gain=1))
        self.i_Hyper = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dim_latent, self.hyperNum), gain=1))
    
        self.rec_iv = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.xavier_uniform_(self.rec_iv.weight)
        self.rec_ia = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.xavier_uniform_(self.rec_ia.weight)
        self.rec_it = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.xavier_uniform_(self.rec_it.weight)
        
        self.rec_uv = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.xavier_uniform_(self.rec_uv.weight)
        self.rec_ua = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.xavier_uniform_(self.rec_ua.weight)        
        self.rec_ut = nn.Linear(self.dim_latent, self.dim_latent)
        nn.init.xavier_uniform_(self.rec_ut.weight)
        
        
        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, self.n_users, self.n_items, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device)  # 256)
        
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, self.n_users, self.n_items, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device)
        if self.a_feat is not None:
            self.a_gcn = GCN(self.dataset, self.n_users, self.n_items, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device)

        self.rec_gcn = GCN(self.dataset, self.n_users, self.n_items, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device)

        
        self.user_hyper_gcn = HyperGNN(keep_rate=self.hyper_keep_rate, layers=self.hyper_layers)
        self.item_hyper_gcn = HyperGNN(keep_rate=self.hyper_keep_rate, layers=self.hyper_layers)
        

        self.ssl_criterion = nn.CrossEntropyLoss()
        self.rec_criterion = nn.MSELoss()

        self.reg_loss = L2Loss()
        self.scl_criterion = nn.KLDivLoss(reduce='batchmean')
        self.log_softmax = nn.LogSoftmax()



    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        self.v_feat = torch.nn.functional.normalize(self.v_feat, dim=1)
        self.a_feat = torch.nn.functional.normalize(self.a_feat, dim=1)
        self.t_feat = torch.nn.functional.normalize(self.t_feat, dim=1)
        v_feat = self.MLP_v(self.v_feat)
        a_feat = self.MLP_a(self.a_feat)
        t_feat = self.MLP_t(self.t_feat)
        
        if self.i_e_agg == "sum":
            v_feat = v_feat + self.i_e
            a_feat = a_feat + self.i_e
            t_feat = t_feat + self.i_e
        

        if self.v_feat is not None:
            v_rep = self.v_gcn(self.edge_index, self.u_e, v_feat)
        
        if self.t_feat is not None:
            t_rep  = self.t_gcn(self.edge_index, self.u_e, t_feat)

        if self.a_feat is not None:
            a_rep = self.a_gcn(self.edge_index, self.u_e, a_feat)
    
        v_item_rep = v_rep[self.n_users:]
        v_user_rep = v_rep[:self.n_users]
        v_e = torch.cat((v_user_rep, v_item_rep), dim=0)
        self.pos_v_tensor = v_e[pos_item_nodes]
        self.neg_v_tensor = v_e[neg_item_nodes]
        self.user_v = v_e[user_nodes]
        # self.hyper_user_v = self.user_hyper_gcn(self.user_v @ self.u_Hyper, self.user_v)
        # self.hyper_pos_v_tensor = self.item_hyper_gcn(self.pos_v_tensor @ self.i_Hyper, self.pos_v_tensor)

        a_item_rep = a_rep[self.n_users:]
        a_user_rep = a_rep[:self.n_users]
        a_e = torch.cat((a_user_rep,a_item_rep), dim=0)
        self.pos_a_tensor = a_e[pos_item_nodes]
        self.neg_a_tensor = a_e[neg_item_nodes]
        self.user_a = a_e[user_nodes]
        # self.hyper_user_a = self.user_hyper_gcn(self.user_a @ self.u_Hyper, self.user_a)
        # self.hyper_pos_a_tensor = self.item_hyper_gcn(self.pos_a_tensor @ self.i_Hyper, self.pos_a_tensor)



        t_item_rep = t_rep[self.n_users:]
        t_user_rep = t_rep[:self.n_users]
        t_e = torch.cat((t_user_rep, t_item_rep), dim=0)
        self.pos_t_tensor = t_e[pos_item_nodes]
        self.neg_t_tensor = t_e[neg_item_nodes]
        self.user_t = t_e[user_nodes]
        # self.hyper_user_t = self.user_hyper_gcn(self.user_t @ self.u_Hyper, self.user_t)
        # self.hyper_pos_t_tensor = self.item_hyper_gcn(self.pos_t_tensor @ self.i_Hyper, self.pos_t_tensor)


        self._momentum_update_tuning()
        rec_rep = self.rec_gcn(self.edge_index, self.u_e, self.i_e)
        rec_item_rep = rec_rep[self.n_users:]
        rec_user_rep = rec_rep[:self.n_users]
        self.item_e = (rec_item_rep + v_item_rep + a_item_rep + t_item_rep) / 4.
        self.user_e = (rec_user_rep + v_user_rep + a_user_rep + t_user_rep) / 4.
        
        self.result_emb = torch.cat((rec_user_rep, rec_item_rep), dim=0)

        pos_item_tensor = self.result_emb[pos_item_nodes]
        neg_item_tensor = self.result_emb[neg_item_nodes]
        user_tensor = self.result_emb[user_nodes]
        

        
        self.rec_item_ev = self.rec_iv(user_tensor)
        self.rec_item_ea = self.rec_ia(user_tensor)
        self.rec_item_et = self.rec_it(user_tensor)
        self.rec_user_ev = self.rec_uv(pos_item_tensor)
        self.rec_user_ea = self.rec_ua(pos_item_tensor)
        self.rec_user_et = self.rec_ut(pos_item_tensor)

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores
        

    @torch.no_grad()
    def _momentum_update_tuning(self):
        for param_v, param_a, param_t, param_rec in zip(
            self.v_gcn.parameters(), self.a_gcn.parameters(), self.t_gcn.parameters(), self.rec_gcn.parameters()
        ):
            # import pdb; pdb.set_trace()
            param_rec.data = (param_v.data + param_a.data + param_t.data)*(1.0 - self.m)/3. + param_rec.data * self.m
            
    def sim(self, z1, z2):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def contrastive_loss(self, z1, z2):
        device = z1.device
        f = lambda x: torch.exp(x / self.tau)   
        
        logits = f(self.sim(z1, z2))
        labels = torch.eye(*logits.shape).to(device)
        return self.ssl_criterion(logits, labels)
    
        # labels = torch.eye(*logits.shape).to(device)
        # x_norm = torch.sqrt(logits)
        # adj = labels.mul(torch.div(x_norm,  torch.trace(x_norm)))
        # rows = adj.sum(dim=1, keepdim=True) 
        # rows[rows==0] = 1
        # adj = adj / rows
        
        
        # return self.scl_criterion(self.log_softmax(logits), adj.softmax(dim=-1))


    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        
        # bpr loss
        pos_scores, neg_scores = self.forward(interaction.clone())
        bpr_loss = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        # emb loss
        u_emb = self.u_e[user]
        i_pos_emb = self.i_e[pos_item]
        i_neg_emb = self.i_e[neg_item]
        # reg_loss = self.reg_loss(u_emb, i_pos_emb, i_neg_emb)
        l2_loss = torch.sum(u_emb**2)*0.5
        l2_loss +=torch.sum(i_pos_emb**2)*0.5
        l2_loss +=torch.sum(i_neg_emb**2)*0.5

        reg_loss = self.reg_weight * l2_loss

        # item_va = self.contrastive_loss(self.hyper_pos_v_tensor, self.hyper_pos_a_tensor)
        # item_vt = self.contrastive_loss(self.hyper_pos_v_tensor, self.hyper_pos_t_tensor)
        # item_at = self.contrastive_loss(self.hyper_pos_a_tensor, self.hyper_pos_t_tensor)
        # user_va = self.contrastive_loss(self.hyper_user_v , self.hyper_user_a)
        # user_vt = self.contrastive_loss(self.hyper_user_v , self.hyper_user_t)
        # user_at = self.contrastive_loss(self.hyper_user_a , self.hyper_user_t)

        item_va = self.contrastive_loss(self.pos_v_tensor, self.pos_a_tensor)
        item_vt = self.contrastive_loss(self.pos_v_tensor, self.pos_t_tensor)
        item_at = self.contrastive_loss(self.pos_a_tensor, self.pos_t_tensor)
        user_va = self.contrastive_loss(self.user_v , self.user_a)
        user_vt = self.contrastive_loss(self.user_v , self.user_t)
        user_at = self.contrastive_loss(self.user_a , self.user_t)

        # l2 l2
        rec_loss = self.rec_criterion(self.rec_user_ev, self.user_v) + self.rec_criterion(self.rec_user_ea, self.user_a) + self.rec_criterion(self.rec_user_et, self.user_t) 
        + self.rec_criterion(self.rec_item_ev, self.pos_v_tensor) + self.rec_criterion(self.rec_item_ea, self.pos_a_tensor) + self.rec_criterion(self.rec_item_et, self.pos_t_tensor)
        
        return bpr_loss + reg_loss + rec_loss + self.cl_weight * (item_va + item_vt + item_at + user_va + user_vt + user_at)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_tensor = self.result_emb[:self.n_users]
        item_tensor = self.result_emb[self.n_users:]

        temp_user_tensor = user_tensor[user, :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features,user_graph,user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix,u_features)
        u_pre = u_pre.squeeze()
        return u_pre

class HyperGNN(nn.Module):
    def __init__(self, keep_rate, layers):
        super().__init__()
        self.keep_rate = keep_rate
        self.layers = layers
    def forward(self, adj, embeds):

        adj = F.dropout(adj, 1-self.keep_rate)
        lat = (adj.T @ embeds)
        ret = (adj @ lat)
        # ret = lat
        return ret

class GCN(torch.nn.Module):
    def __init__(self,datasets, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None,device = None,features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id       
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        self.gcns = nn.ModuleList([Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode) for _ in range(self.num_layer)])
        # self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)


    def forward(self,edge_index, user, item):
        # temp_features = self.MLP_1(F.leaky_relu(self.MLP(item))) if self.dim_latent else item
        # item = F.normalize(item)
        x = torch.cat((user, item), dim=0).to(self.device)
        # h = self.conv_embed_1(x, edge_index)  # equation 1
        # h_1 = self.conv_embed_1(h, edge_index)
        for gcn in self.gcns:
            x = gcn(x, edge_index) + x
        # x_hat = h + x + h_1
        return x


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


