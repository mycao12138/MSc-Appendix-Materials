import os
import numpy as np
import random
import torch
from random import shuffle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, k_hop_subgraph

seed = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


def __seeds_list__(nodes):
    split_size = max(5, int(nodes.shape[0] / 400))
    seeds_list = list(torch.split(nodes, split_size))
    if len(seeds_list) < 400:
        print('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
        seeds_list = []
        while len(seeds_list) < 400:
            split_size = random.randint(3, 5)
            seeds_list_1 = torch.split(nodes, split_size)
            seeds_list = seeds_list + list(seeds_list_1)
            nodes = nodes[torch.randperm(nodes.shape[0])]
    shuffle(seeds_list)
    seeds_list = seeds_list[0:400]

    return seeds_list


def __dname__(p, task_id):
    if p == 0:
        dname = 'task{}.meta.train.support'.format(task_id)
    elif p == 1:
        dname = 'task{}.meta.train.query'.format(task_id)
    elif p == 2:
        dname = 'task{}.meta.test.support'.format(task_id)
    elif p == 3:
        dname = 'task{}.meta.test.query'.format(task_id)
    else:
        raise KeyError

    return dname


def __pos_neg_nodes__(labeled_nodes, node_labels, i: int):
    pos_nodes = labeled_nodes[node_labels[:, i] == 1]
    pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
    neg_nodes = labeled_nodes[node_labels[:, i] == 0]
    neg_nodes = neg_nodes[torch.randperm(neg_nodes.shape[0])]
    return pos_nodes, neg_nodes


def __induced_graph_list_for_graphs__(seeds_list, label, p, num_nodes, potential_nodes, ori_x, same_label_edge_index,
                                      smallest_size, largest_size):
    seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
    induced_graph_list = []
    for seeds in seeds_part_list:

        subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=1, num_nodes=num_nodes,
                                         edge_index=same_label_edge_index, relabel_nodes=True)

        # regularize its size

        temp_hop = 1
        while len(subset) < smallest_size and temp_hop < 5:
            temp_hop = temp_hop + 1
            subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=temp_hop, num_nodes=num_nodes,
                                             edge_index=same_label_edge_index, relabel_nodes=True)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.from_numpy(np.setdiff1d(potential_nodes.numpy(), subset.numpy()))

            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            # directly downmsample
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
            subset = torch.unique(torch.cat([torch.flatten(seeds), subset]))

        sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

        x = ori_x[subset]
        graph = Data(x=x, edge_index=sub_edge_index, y=label)
        induced_graph_list.append(graph)

    return induced_graph_list


def graph_views(data, aug='random', aug_ratio=0.1):
    """
    each for a single graph, not a batch graphs!
    :param data:
    :param aug:
    :param aug_ratio:
    :return:
    """

    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)

    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data = drop_nodes(data, aug_ratio)
        elif n == 1:
            data = permute_edges(data, aug_ratio)
        else:
            print('augmentation error')
            assert False
    return data


def drop_nodes(data, aug_ratio):
    """
    only responsible for data.x, data.edge_index !
    all the other keys might be unconsistent!
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
    except:
        data = data

    return data


def permute_edges(data, aug_ratio):
    """
    only change edge_index, all the other keys unchanged and consistent
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]

    return data


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token.clone().detach()

    return data


def get_loader(graph_list, batch_size, aug1=None, aug2=None, aug_ratio=None):
    """
    this function is used for pre-training components
    """
    if len(graph_list) % batch_size == 1:
        raise KeyError(
            "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

    shuffle(graph_list)
    if aug1 is None:
        aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
    if aug2 is None:
        aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
    if aug_ratio is None:
        aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

    print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

    view_list_1 = []
    view_list_2 = []
    for g in graph_list:
        view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
        view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
        view_list_1.append(view_g)
        # for g in graph_list_2:
        view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
        view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
        view_list_2.append(view_g)

    loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                         num_workers=1)  # you must set shuffle=False !
    loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                         num_workers=1)  # you must set shuffle=False !

    return loader1, loader2

