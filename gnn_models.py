import math
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
torch.manual_seed(3) 
import scipy
import dgl.nn.pytorch.conv as dglnn
from torch import nn
from scipy.special import comb
import math
np.random.seed(1)
import dgl.function as fn
import dgl
import dgl.nn.pytorch.conv as dglnn
from torch import nn
from dgl.nn import AvgPooling,MaxPooling,WeightAndSum,GlobalAttentionPooling,SortPooling
from utils import *
from torch.autograd import Variable
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def log(*pargs):
    with open('/apdcephfs/share_1364275/kaithgao/RLNDOCK_CODE/my_log/inference.txt', 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")
 
class GNN_oracle(nn.Module):
    def __init__(self, in_feats=13, h_feats=32,cls_h = 256, num_layers=4, dropout_rate=0.,
                 activation='ELU',GNN='gin',pooling = 'mean'):
        super().__init__()
        self.cls1 = nn.Linear(13, cls_h)
        self.cls2 = nn.Linear(cls_h, 1)
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.lm = nn.LayerNorm(h_feats)
        self.lm_last = nn.LayerNorm(13)
        if GNN=='gcn':
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act, allow_zero_in_degree=True))
            for i in range(num_layers-1):
                self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(h_feats, 13, activation=self.act, allow_zero_in_degree=True))
        elif GNN=='gin':
            self.layers.append(dglnn.GINConv(torch.nn.Linear(h_feats,h_feats),'max',learn_eps=True))
            for i in range(num_layers-1):
                self.layers.append(dglnn.GINConv(torch.nn.Linear(h_feats,h_feats),'max',learn_eps=True))
            self.layers.append(dglnn.GINConv(torch.nn.Linear(h_feats,13),'max',learn_eps=True))
        self.mlp_before = MLP_before(in_feats, h_feats, dropout_rate)
        self.mlp = MLP_oracle(h_feats, h_feats, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if pooling == 'mean':
            self.pool = AvgPooling()
        elif pooling == 'max':
            self.pool = MaxPooling()
        elif pooling == 'weight':
            self.pool = WeightAndSum(64)
        elif pooling == 'sort':
            self.pool = SortPooling(k=2)
        elif pooling == 'attention':
            self.pool = GlobalAttentionPooling(torch.nn.Linear(64, 1))

    def reset_parameters(self):
        if self.layer_1.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_1.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)
        if self.layer_2.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_2.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)

    def forward(self, graph, edge_weight = None):
        # h = torch.cat((graph.ndata['mu_r_norm'],graph.ndata['res_feat']),dim=1)
        h = self.mlp_before(graph.ndata['features'])
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            if i != len(self.layers)-1:
                h = self.lm(layer(graph, h, edge_weight = edge_weight))
            else:
                h = self.lm_last(layer(graph, h, edge_weight = edge_weight))
        nodes_emb = h.clone()
        h = self.pool(graph, h)
        # h = self.pool(graph, self.mlp(h))
        return self.cls2(self.act(self.cls1(h))), nodes_emb
    
    
    def forward_action(self, edge_index_all, edge_weight_all, feature_matrix_all):
        g = dgl.DGLGraph((np.array(edge_index_all).tolist()[0],np.array(edge_index_all).tolist()[1])).to(device)
        g.ndata['features'] = feature_matrix_all                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        # pred_rmsd = self.forward(dgl.add_self_loop(g), edge_weight = edge_weight_all)
        pred_rmsd,_ = self.forward(g, edge_weight = edge_weight_all)
        return pred_rmsd
    
    def update_complex(self, complex_rep_clone, edge_index_all, edge_weight_init, feature_matrix_init):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        N_now, feat_dim = feature_matrix_init.shape[0], feature_matrix_init.shape[1]
        
        complex_rep_clone = complex_rep_clone.to(device)
        edge_weight_init = Variable(edge_weight_init, requires_grad = False)
        edge_weight_expend = Variable(torch.rand(N_now).to(device), requires_grad = True)
        feature_matrix_expend = Variable(torch.softmax(torch.rand((1,complex_rep_clone.shape[0])),0).to(device), requires_grad = True)
        
        complex_rep_clone = Variable(complex_rep_clone, requires_grad = False)

        # opt_action = torch.optim.Adam([edge_weight_expend, feature_matrix_expend], lr=0.01, betas=(0.9, 0.99))
        opt_action = torch.optim.Adam(
            [
                {'params': edge_weight_expend,'lr': 1e-2},
                {'params': feature_matrix_expend, 'lr': 1e-1}
            ]
        )
                
        for i in range(200): ### 5k 10k
            edge_weight_all = torch.cat((edge_weight_init, torch.softmax(edge_weight_expend.to(device),0)),dim=0)
            # edge_weight_all = torch.cat((edge_weight_init, edge_weight_expend.to(device)),dim=0)
            # feature_matrix_all = torch.cat((feature_matrix_init, (torch.softmax(feature_matrix_expend,0) @ complex_rep_clone).to(device)),dim=0)  
            feature_matrix_all = torch.cat((feature_matrix_init, (torch.softmax(feature_matrix_expend,1) @ complex_rep_clone).to(device)),dim=0)      
            pred_rmsd = self.forward_action(edge_index_all, edge_weight_all, feature_matrix_all).abs()
            if i % 20 ==0:
                print(np.array(pred_rmsd.detach().cpu()))
            opt_action.zero_grad()
            pred_rmsd.backward()
            opt_action.step()
            # print(torch.softmax(feature_matrix_expend,1))
            # print(edge_weight_expend)

        

        # return edge_weight_expend, f_exp 
        return edge_weight_expend, feature_matrix_expend @ complex_rep_clone
class MLP_before(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate=0.2, activation='ELU'):
        super(MLP_before, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        ra = 4
        self.layers_1 = nn.Linear(in_feats, h_feats*ra)
        self.layers_2 = nn.Linear(h_feats*ra, h_feats)
        self.lm_1 = nn.LayerNorm(h_feats*ra)
        self.lm_2 = nn.LayerNorm(h_feats)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def reset_parameters(self):
        if self.layer_1.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_1.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)
        if self.layer_2.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_2.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)

    def forward(self, h):
        h = self.dropout(self.lm_1(self.act(self.layers_1(h))))
        h = self.dropout(self.lm_2(self.act(self.layers_2(h))))
        return h   

class MLP_oracle(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate=0.2, activation='ELU'):
        super(MLP_oracle, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers_1 = nn.Linear(in_feats, h_feats)
        self.layers_2 = nn.Linear(h_feats, h_feats)
        self.lm = nn.LayerNorm(h_feats)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def reset_parameters(self):
        if self.layer_1.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_1.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)
        if self.layer_2.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_2.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)

    def forward(self, h):
        h = self.lm(self.act(self.layers_1(h)))
        h = self.dropout(h)
        h = self.lm(self.act(self.layers_2(h)))
        h = self.dropout(h)
        return h



class MLP_oracle_BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate=0.2, activation='ELU'):
        super(MLP_oracle_BWGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers_1 = nn.Linear(in_feats*4, h_feats)
        self.layers_2 = nn.Linear(h_feats, h_feats)
        self.lm = nn.LayerNorm(h_feats)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def reset_parameters(self):
        if self.layer_1.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_1.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)
        if self.layer_2.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_2.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)

    def forward(self, h):
        h = self.dropout(h)
        h = self.lm(self.act(self.layers_1(h)))
        h = self.dropout(h)
        h = self.lm(self.act(self.layers_2(h)))
        h = self.dropout(h)
        h = self.lm(self.act(self.layers_2(h)))
        return h
    
class target_model(nn.Module):
    def __init__(self, in_feats, h_feats, f_feats, dropout_rate=0.2, activation='ELU'):
        super(target_model, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers_1 = nn.Linear(in_feats, h_feats)
        self.layers_2 = nn.Linear(h_feats, h_feats)
        self.layers_3 = nn.Linear(h_feats, f_feats)
        self.layers_4 = nn.Linear(h_feats, h_feats)
        self.layers_5 = nn.Linear(h_feats, f_feats)
        self.lm1 = nn.LayerNorm(h_feats)
        self.lm2 = nn.LayerNorm(h_feats)
        self.lm3 = nn.LayerNorm(f_feats)
        self.lm4 = nn.LayerNorm(h_feats)
        self.lm5 = nn.LayerNorm(f_feats)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.c_a = CrossAttentionLayer(13,13)

    def reset_parameters(self):
        if self.layer_1.weight is not None:
            nn.init.xavier_normal_(self.layers_1.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_1.bias is not None:
            nn.init.constant_(self.layer_1.bias.data, 0.0)
        if self.layer_2.weight is not None:
            nn.init.xavier_normal_(self.layers_2.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_2.bias is not None:
            nn.init.constant_(self.layer_2.bias.data, 0.0)
        if self.layer_3.weight is not None:
            nn.init.xavier_normal_(self.layers_3.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_3.bias is not None:
            nn.init.constant_(self.layer_3.bias.data, 0.0)
        if self.layer_4.weight is not None:
            nn.init.xavier_normal_(self.layers_4.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_4.bias is not None:
            nn.init.constant_(self.layer_4.bias.data, 0.0)
        if self.layer_5.weight is not None:
            nn.init.xavier_normal_(self.layers_5.weight, gain=0.02) # Implement Xavier Uniform
        if self.layer_5.bias is not None:
            nn.init.constant_(self.layer_5.bias.data, 0.0)

    def forward(self, h):
        # h1 = self.c_a(h[0,:].unsqueeze(0), h[1,:].unsqueeze(0), [1], [1])
        # h0 = self.c_a(h[1,:].unsqueeze(0), h[0,:].unsqueeze(0), [1], [1])
        # h = torch.cat((h0,h1),0)
        h_clone = h.clone()
        for b in range(int(h.size(0)/2)):
            h_clone[[b*2,b*2+1]] = h_clone[[b*2+1,b*2]]
        h = torch.cat((h,h_clone),1)
        h = self.dropout(h)
        h = self.lm1(self.act(self.layers_1(h)))
        h = self.dropout(h)
        h = self.lm2(self.act(self.layers_2(h)))
        h = self.dropout(h)
        h = self.lm3(self.act(self.layers_3(h)))
        # h = self.lm4(self.act(self.layers_4(h)))
        # h = self.lm5(self.act(self.layers_5(h)))

        return self.act(h)
    

class CrossAttentionLayer(nn.Module):
    def __init__(self,h_dim,num_heads):
        super(CrossAttentionLayer,self).__init__()
        self.h_dim = h_dim
        self.h_dim_div = self.h_dim // 1
        self.num_heads = num_heads
        assert self.h_dim_div % self.num_heads == 0
        self.head_dim = self.h_dim_div // self.num_heads
        self.merge = nn.Conv1d(self.h_dim_div, self.h_dim_div, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(self.h_dim, self.h_dim_div, kernel_size=1) for _ in range(3)])
        dropout = 0.

        self.mlp = nn.Sequential(
            nn.Linear(self.h_dim+self.h_dim_div, self.h_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(self.h_dim)
        )
    
    def forward(self, src_h, dst_h, src_num_verts, dst_num_verts):
        h = dst_h
        src_h_list = torch.split(src_h, src_num_verts)
        dst_h_list = torch.split(dst_h, dst_num_verts)
        h_msg = []
        for idx in range(len(src_num_verts)):
            src_hh = src_h_list[idx].unsqueeze(0).transpose(1, 2)
            dst_hh = dst_h_list[idx].unsqueeze(0).transpose(1, 2)
            query, key, value = [hh.view(1, self.head_dim, self.num_heads, -1) \
                for ll, hh in zip(self.proj, (dst_hh, src_hh, src_hh))]
            dim = query.shape[1]
            scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / (dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h_dst = torch.einsum('bhnm,bdhm->bdhn', attn, value) 
            h_dst = h_dst.contiguous().view(1, self.h_dim_div, -1)
            h_msg.append(h_dst.squeeze(0).transpose(0, 1))
        h_msg = torch.cat(h_msg, dim=0)

        # skip connection
        h_out = h + self.mlp(torch.cat((h, h_msg), dim=-1))

        return h_out