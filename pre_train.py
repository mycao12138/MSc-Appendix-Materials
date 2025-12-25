import torch
import torch.optim as optim
from torch_geometric.loader.cluster import ClusterData
from torch.autograd import Variable
from MPG.Models import GCN, GraphCL
from utils import get_loader
from copy import deepcopy
from torch_geometric.loader import DataLoader
import pickle as pk
from utils import mkdir
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data


def gen_ran_output(data, model):
    vice_model = deepcopy(model)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)

    return z2


def train_simgrace(model, loader, optimizer):
    model.train()
    train_loss_accum = 0
    total_step = 0
    for step, data in enumerate(loader):
        optimizer.zero_grad()
        x2 = gen_ran_output(data, model)
        x1 = model.forward_cl(data.x, data.edge_index, data.batch)
        x2 = Variable(x2.detach().data, requires_grad=False)
        loss = model.loss_cl(x1, x2)
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())
        total_step = total_step + 1

    return train_loss_accum / total_step


def train_graphcl(model, loader1, loader2, optimizer):
    model.train()
    train_loss_accum = 0
    total_step = 0
    for step, batch in enumerate(zip(loader1, loader2)):
        batch1, batch2 = batch
        optimizer.zero_grad()
        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        total_step = total_step + 1

    return train_loss_accum / total_step


if __name__ == '__main__':
    # load data
    mkdir('./pre_trained_gnn/')
    dataname = 'personalitycafe'
    data = pk.load(open('./Dataset/{}/{}.data'.format(dataname, dataname), 'br'))
    num_parts = 200
    batch_size = 10

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)

    graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='./Dataset/{}/'.format(dataname)))

    lr, decay, epochs = 0.01, 0.0001, 100
    input_dim = data.x.shape[1]

    hid_dim = input_dim

    for pre_train_method in ['GraphCL', 'SimGRACE']:
        if pre_train_method == 'GraphCL':
            loader1, loader2 = get_loader(graph_list, batch_size, aug1='dropN', aug2="permE")
        elif pre_train_method == 'SimGRACE':
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)

        for gnn_type in ['TransformerConv', 'GAT', 'GCN']:
            gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, pool=None,
                      gnn_type=gnn_type)
            model = GraphCL(gnn, hid_dim=hid_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

            print('start training {} | {} | {}...'.format(dataname, pre_train_method, gnn_type))
            train_loss_min = 1000000
            for epoch in range(1, epochs + 1):  # 1..100
                if pre_train_method == 'GraphCL':
                    train_loss = train_graphcl(model, loader1, loader2, optimizer)
                elif pre_train_method == 'SimGRACE':
                    train_loss = train_simgrace(model, loader, optimizer)
                print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

                if train_loss_min > train_loss:
                    train_loss_min = train_loss
                    torch.save(model.gnn.state_dict(),
                               "./pre_trained_gnn/{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type))
                    print("+++model saved ! {}.{}.{}.pth".format(dataname, pre_train_method, gnn_type))
