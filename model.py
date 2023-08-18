"""model.py"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add


# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModel(torch.nn.Module):
    def __init__(self, args, dims):
        super(EdgeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.edge_mlp = MLP([3*self.dim_hidden + dims['g']] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        if u is not None:
            out = torch.cat([edge_attr, src, dest, u[batch]], dim=1)
        else: 
            out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, args, dims):
        super(NodeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.node_mlp = MLP([2*self.dim_hidden + dims['f'] + dims['g']] + self.n_hidden*[self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):
        src, dest = edge_index
        out = scatter_add(edge_attr, dest, dim=0, dim_size=x.size(0))
        if f is not None:
            out = torch.cat([x, out, f], dim=1)
        elif u is not None:
            out = torch.cat([x, out, u[batch]], dim=1)
        else:
            out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)
        return out


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):

        src = edge_index[0]
        dest = edge_index[1]

        edge_attr = self.edge_model(x[src], x[dest], edge_attr, u,
                                    batch if batch is None else batch[src])
        x = self.node_model(x, edge_index, edge_attr, f, u, batch)

        return x, edge_attr


# Thermodyncamics-informed Graph Neural Networks
class TIGNN(torch.nn.Module):
    def __init__(self, args, dims):
        super(TIGNN, self).__init__()
        # Arguments
        passes = args.passes
        n_hidden = args.n_hidden
        dim_hidden = args.dim_hidden
        self.dims = dims
        self.dim_z = self.dims['z']
        self.dim_q = self.dims['q']
        dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden*[dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden*[dim_hidden] + [dim_hidden])
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(args, self.dims)
            edge_model = EdgeModel(args, self.dims)
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)
        # Decoder MLPs
        self.decoder_E = MLP([dim_hidden] + n_hidden*[dim_hidden] + [1])
        self.decoder_S = MLP([dim_hidden] + n_hidden*[dim_hidden] + [1])
        self.decoder_L = MLP([dim_hidden] + n_hidden*[dim_hidden] + [int(self.dim_z*(self.dim_z+1)/2-self.dim_z)])
        self.decoder_M = MLP([dim_hidden] + n_hidden*[dim_hidden] + [int(self.dim_z*(self.dim_z+1)/2)])

        diag = torch.eye(self.dim_z, self.dim_z)
        self.diag = diag[None]
        self.ones = torch.ones(self.dim_z, self.dim_z)

    def forward(self, z, n, edge_index, q_0=None, f=None, g=None, batch=None): 
        '''Pre-process'''
        z.requires_grad = True
        # Node attributes 
        # Eulerian
        if q_0 is not None:
            q = q_0
            v = z
        # Lagrangian
        else:
            q = z[:,:self.dim_q]
            v = z[:,self.dim_q:]
        x = torch.cat((v,n), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u,dim=1).reshape(-1,1)
        edge_attr = torch.cat((u,u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=f, u=g, batch=batch)
            x += x_res
            edge_attr += edge_attr_res

        '''Decode'''
        # Gradients
        E = self.decoder_E(x)
        S = self.decoder_S(x)
        dEdz = torch.autograd.grad(E, z, torch.ones(E.shape, device=E.device), create_graph=True)[0]
        dSdz = torch.autograd.grad(S, z, torch.ones(S.shape, device=S.device), create_graph=True)[0]
        # GENERIC flattened matrices
        l = self.decoder_L(x)
        m = self.decoder_M(x)

        '''Reparametrization'''
        L = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=l.device)
        M = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=m.device)
        L[:,torch.tril(self.ones,-1) == 1] = l
        M[:,torch.tril(self.ones) == 1] = m
        # L skew-symmetric
        L = L - torch.transpose(L,1,2)
        # M symmetric and positive semi-definite
        M = torch.bmm(M,torch.transpose(M,1,2))

        return L, M, dEdz.unsqueeze(2), dSdz.unsqueeze(2), E, S


if __name__ == '__main__':
    pass
