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

        self.MLP_v = nn.Linear(128, self.dim_latent, bias=False)
        self.MLP_a = nn.Linear(128, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(64, self.dim_latent, bias=False)


        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']), allow_pickle=True).item()

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

    
        self.u_e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.dim_latent)))
        self.i_e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.dim_latent)))
       

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

        self.ssl_criterion = nn.CrossEntropyLoss()

        self.reg_loss = L2Loss()



    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
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

        a_item_rep = a_rep[self.n_users:]
        a_user_rep = a_rep[:self.n_users]
        a_e = torch.cat((a_user_rep,a_item_rep), dim=0)
        self.pos_a_tensor = a_e[pos_item_nodes]
        self.neg_a_tensor = a_e[neg_item_nodes]
        self.user_a = a_e[user_nodes]



        t_item_rep = t_rep[self.n_users:]
        t_user_rep = t_rep[:self.n_users]
        t_e = torch.cat((t_user_rep, t_item_rep), dim=0)
        self.pos_t_tensor = t_e[pos_item_nodes]
        self.neg_t_tensor = t_e[neg_item_nodes]
        self.user_t = t_e[user_nodes]


        self._momentum_update_tuning()
        rec_rep = self.rec_gcn(self.edge_index, self.u_e, self.i_e)
        rec_item_rep = rec_rep[self.n_users:]
        rec_user_rep = rec_rep[:self.n_users]
        item_e = (rec_item_rep + v_item_rep + a_item_rep + t_item_rep) / 4.
        user_e = (rec_user_rep + v_user_rep + a_user_rep + t_user_rep) / 4.
        self.result_emb = torch.cat((user_e, item_e), dim=0)

        pos_item_tensor = self.result_emb[pos_item_nodes]
        neg_item_tensor = self.result_emb[neg_item_nodes]
        user_tensor = self.result_emb[user_nodes]

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
        
        labels  = torch.tensor(list(range(z1.shape[0]))).to(device)
        logits = f(self.sim(z1, z2))
        return self.ssl_criterion(logits, labels)


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

        item_va = self.contrastive_loss(self.pos_v_tensor, self.pos_a_tensor)
        item_vt = self.contrastive_loss(self.pos_v_tensor, self.pos_t_tensor)
        item_at = self.contrastive_loss(self.pos_a_tensor, self.pos_t_tensor)
        user_va = self.contrastive_loss(self.user_v , self.user_a )
        user_vt = self.contrastive_loss(self.user_v , self.user_t )
        user_at = self.contrastive_loss(self.user_a , self.user_t )
        
        return bpr_loss + reg_loss + self.cl_weight * (item_va + item_vt + item_at + user_va + user_vt + user_at)

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
        x = F.normalize(x).to(self.device)
        # h = self.conv_embed_1(x, edge_index)  # equation 1
        # h_1 = self.conv_embed_1(h, edge_index)
        for gcn in self.gcns:
            x_ = gcn(x, edge_index)
            x = x + x_
        # x_hat = h + x + h_1
        return x


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
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

