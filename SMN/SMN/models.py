from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from torch.autograd import Variable
import torch.nn as nn
import os
import math
import argparse
from torch_sparse import spmm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import d2lzh_pytorch as d2l
from torch import sigmoid
import scipy.sparse as sp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.avg_pool import avg_pool
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
        self.flatten = d2l.FlattenLayer()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)
    def forward(self, input):
        return self.linear2(self.relu(self.linear1(self.flatten(input))))


class MaskLinear(Module):
    def __init__(self, in_features, out_features=1, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, masks):  # idx is a list
        # mask = torch.zeros(self.in_features).cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output0 = torch.matmul(self.weight.to(device), torch.mul(masks[0].reshape(-1,1).to(device), x.to(device)))
        for i in range(1, len(masks)):
            mask = torch.mul(masks[i].reshape(-1, 1).to(device), x.to(device))
            output = torch.matmul(self.weight.to(device), mask.to(device))
            lambda_text = torch.vstack((output0, output.to(device)))
            output0 = lambda_text
        if self.bias is not None:
            return lambda_text + self.bias.to(device)
        else:
            return lambda_text

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' => ' \
               + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        # 这里叶子节点
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)

        if self.use_bias:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=100, dropout=0.2):
        super(GcnNet, self).__init__()
        self.dropout = dropout

        self.conv1 = GraphConvolution(input_dim, input_dim)

        self.conv2 = GraphConvolution(input_dim, input_dim)
        self.conv3 = GraphConvolution(input_dim, input_dim)
        # self.conv4 = GraphConvolution(50, 50)
        self.conv5 = GraphConvolution(input_dim, input_dim)

        self.conv_bn = nn.BatchNorm1d(input_dim)
        # self.conv_bn = nn.LayerNorm(input_dim)

        self.pool_train = SAGPool(input_dim, ratio=0.2518)
        self.pool_test = SAGPool(input_dim, ratio=0.2183)
        self.lin2 = torch.nn.Linear(input_dim//2, 1)

    def forward(self, adjacency, x, masks):
        H = F.gelu(self.conv1(adjacency, x))
        H = self.conv_bn(H)

        # H = F.relu(self.conv2(adjacency, H))
        # H = self.conv2_bn(H)
        # H = F.relu(self.conv3(adjacency, H))
        # H = self.conv3_bn(H)
        H = F.gelu(self.conv5(adjacency, H))
        # H = F.dropout(H, p=self.dropout, training=self.training)
        # H = self.conv_bn(H)

        if len(H) > 2000:
            H_p = self.pool_train(H, adjacency, None, None)
            # H_p = F.gelu(H_p)
        else:
            H_p = self.pool_test(H, adjacency, None, None)
            # H_p = F.gelu(H_p)

        return H, H_p

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)  # 生成形状和e相同的全接近0的矩阵
        attention = torch.where(adj > 0, e, zero_vec)  # 按条件取元素
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, ofeat, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, ofeat, dropout=dropout, alpha=alpha, concat=False)
        self.pool_train = SAGPool(100, ratio=0.2518)
        self.pool_test = SAGPool(100, ratio=0.2183)
        # self.mask_train = MaskLinear(3177, 1)
        # self.mask_test = MaskLinear(916, 1)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        if len(x) > 2000:
            x_p = self.pool_train(x, adj, None, None)
        else:
            x_p = self.pool_test(x, adj, None, None)
        return x, x_p


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio, Conv=GraphConvolution, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0)).int().long()
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = sigmoid(self.score_layer(edge_index, x).squeeze())

        perm = topk(score, self.ratio, batch)  # topk选取得分前50%的节点index
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        # batch = batch[perm]
        # edge_index, edge_attr = filter_adj(
        #     edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x
