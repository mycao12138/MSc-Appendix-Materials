#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.utils import sort_edge_index, add_self_loops, to_undirected
from torch_geometric.data import Batch
from utils import seed_everything, seed
from torch_geometric.loader.cluster import ClusterData

seed_everything(seed)


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)


class GraphCL(torch.nn.Module):

    def __init__(self, gnn, hid_dim=16):
        super(GraphCL, self).__init__()
        self.gnn = gnn
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hid_dim, hid_dim))

    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean() + 10
        return loss


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


class Prompt(torch.nn.Module):
    def __init__(self, token_dim, token_num, prune_thre=0.9, isolate_tokens=False, inner_prune=None):
        super(Prompt, self).__init__()
        self.prune_thre = prune_thre
        if inner_prune is None:
            self.inner_prune = prune_thre
        else:
            self.inner_prune = inner_prune
        self.isolate_tokens = isolate_tokens
        self.token_x = torch.nn.Parameter(torch.empty(token_num, token_dim))
        self.initial_prompt()

    def initial_prompt(self, init_mode='kaiming_uniform'):
        if init_mode == 'metis':  # metis_num = token_num
            self.initial_prompt_with_metis()
        elif init_mode == 'node_labels':  # label_num = token_num
            self.initial_prompt_with_node_labels()
        elif init_mode == 'orthogonal':
            torch.nn.init.orthogonal_(self.token_x, gain=torch.nn.init.calculate_gain('leaky_relu'))
        elif init_mode == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.token_x, gain=torch.nn.init.calculate_gain('tanh'))
        elif init_mode == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.token_x, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        elif init_mode == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.token_x, nonlinearity='leaky_relu')
        elif init_mode == 'uniform':
            torch.nn.init.uniform_(self.token_x, 0.99, 1.01)
        else:
            raise KeyError("init_mode {} is not defined!".format(init_mode))

    def initial_prompt_with_metis(self, data=None, save_dir=None):
        if data is None:
            raise KeyError("you are calling initial_prompt_with_metis with empty data")
        metis = ClusterData(data=data, num_parts=self.token_x.shape[0], save_dir=save_dir)
        b = Batch.from_data_list(list(metis))
        x = global_mean_pool(b.x, b.batch)
        self.token_x.data = x

    def initial_prompt_with_node_labels(self, data=None):
        x = global_mean_pool(data.x, batch=data.y)
        self.token_x.data = x

    def forward(self, graph_batch: Batch):
        num_tokens = self.token_x.shape[0]
        node_x = graph_batch.x
        num_nodes = node_x.shape[0]
        num_graphs = graph_batch.num_graphs
        node_batch = graph_batch.batch
        token_x_repeat = self.token_x.repeat(num_graphs, 1)
        token_batch = torch.LongTensor([j for j in range(num_graphs) for i in range(num_tokens)])
        batch_one = torch.cat([node_batch, token_batch], dim=0)
        token_batch = torch.LongTensor([j for j in range(num_graphs) for i in range(num_tokens)]) + num_graphs
        batch_two = torch.cat([node_batch, token_batch], dim=0)
        edge_index = graph_batch.edge_index

        token_dot = torch.mm(self.token_x, torch.transpose(node_x, 0, 1))  # (T,d) (d, N)--> (T,N)
        token_sim = torch.sigmoid(token_dot)  # 0-1
        cross_adj = torch.where(token_sim < self.prune_thre, 0, token_sim)
        tokenID2nodeID = cross_adj.nonzero().t().contiguous()
        batch_value = node_batch[tokenID2nodeID[1]]
        new_token_id_in_cross_edge = tokenID2nodeID[0] + num_nodes + num_tokens * batch_value
        tokenID2nodeID[0] = new_token_id_in_cross_edge
        cross_edge_index = tokenID2nodeID
        if self.isolate_tokens:
            new_edge_index = torch.cat([edge_index, cross_edge_index], dim=1)
        else:
            token_dot = torch.mm(self.token_x, torch.transpose(self.token_x, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1
            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            tokenID2tokenID = inner_adj.nonzero().t().contiguous()
            inner_edge_index = torch.cat([tokenID2tokenID + num_nodes + num_tokens * i for i in range(num_graphs)],
                                         dim=1)
            new_edge_index = torch.cat([edge_index, cross_edge_index, inner_edge_index], dim=1)

        new_edge_index, _ = add_self_loops(new_edge_index)
        new_edge_index = to_undirected(new_edge_index)
        edge_index_xp = sort_edge_index(new_edge_index)

        return (torch.cat([node_x, token_x_repeat], dim=0),
                edge_index_xp,
                batch_one,
                batch_two)


class Pipeline(torch.nn.Module):
    def __init__(self, input_dim,
                 pre_train_path=None, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 frozen_gnn='all',
                 with_prompt=True, token_num=5, prune_thre=0.5, inner_prune=None, isolate_tokens=False,
                 frozen_project_head=False, pool_mode=1, gnn_type='GAT', project_head_path=None):

        super().__init__()
        self.with_prompt = with_prompt
        self.pool_mode = pool_mode
        if with_prompt:
            self.token_num = token_num
            self.prompt = Prompt(token_dim=input_dim, token_num=token_num, prune_thre=prune_thre,
                                 isolate_tokens=isolate_tokens, inner_prune=inner_prune)
        self.gnn = GCN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)

        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, num_classes),
            torch.nn.Softmax(dim=1))

        self.set_gnn_project_head(pre_train_path, frozen_gnn, frozen_project_head, project_head_path)

    def set_gnn_project_head(self, pre_train_path, frozen_gnn, frozen_project_head, project_head_path=None):
        if pre_train_path:
            self.gnn.load_state_dict(torch.load(pre_train_path))
            print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))

        if project_head_path:
            self.project_head.load_state_dict(torch.load(project_head_path))
            print("successfully load project_head! @ {}".format(project_head_path))

        if frozen_gnn == 'all':
            for p in self.gnn.parameters():
                p.requires_grad = False
        elif frozen_gnn == 'none':
            for p in self.gnn.parameters():
                p.requires_grad = True
        else:
            pass

        if frozen_project_head:
            for p in self.project_head.parameters():
                p.requires_grad = False

    def forward(self, graph_batch: Batch):
        num_graphs = graph_batch.num_graphs
        if self.with_prompt:
            xp, xp_edge_index, batch_one, batch_two = self.prompt(graph_batch)

            if self.pool_mode == 1:
                graph_emb = self.gnn(xp, xp_edge_index, batch_one)
                pre = self.project_head(graph_emb)
                return pre
            elif self.pool_mode == 2:
                emb = self.gnn(xp, xp_edge_index, batch_two)
                graph_emb = emb[0:num_graphs, :]
                prompt_emb = emb[num_graphs:, :]
                com_emb = graph_emb - prompt_emb
                pre = self.project_head(com_emb)
                return pre
        else:
            graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            pre = self.project_head(graph_emb)
            return pre
