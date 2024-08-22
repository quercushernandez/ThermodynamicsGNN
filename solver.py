"""solver.py"""

import os

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import TIGNN
from dataset import load_dataset
from utilities.plot import plot_2D, plot_3D
from utilities.utils import save_log, print_error


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Study Case
        self.sys_name = args.sys_name
        self.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

        # Dataset Parameters
        self.train_set, self.val_set, self.test_set = load_dataset(args)
        self.dims = self.train_set.dims
        self.dt = self.train_set.dt

        # Normalization
        self.stats_z, self.stats_q = self.train_set.get_stats(self.device)

        # Training Parameters
        self.max_epoch = args.max_epoch
        self.lambda_d = args.lambda_d
        self.batch_size = args.batch_size
        self.noise_var = args.noise_var

        # Net Parameters
        self.net = TIGNN(args, self.dims).to(self.device).float() 
        if (args.train == False):
            # Load pretrained net
            load_name = 'pretrained_' + self.sys_name + '.pt'
            load_path = os.path.join(args.dset_dir, load_name)
            checkpoint = torch.load(load_path, map_location=self.device)
            self.net.load_state_dict(checkpoint)

        self.optim = optim.Adam(self.net.parameters(), lr=args.lr) 
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.miles, gamma=args.gamma)

        # Load/Save options
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)


    def train_model(self):
        epoch = 0
        train_log = {'epoch':[], 'loss_z':[], 'loss_deg_E':[], 'loss_deg_S':[]}
        val_log = {'epoch':[], 'loss_z':[], 'loss_deg_E':[], 'loss_deg_S':[]}

        print("\n[Training Started]\n")
        
        # Main training loop
        while (epoch < self.max_epoch):
            print('[Epoch: {}]'.format(epoch+1))

            # Train set loop
            loss_z_sum = 0
            loss_deg_E_sum, loss_deg_S_sum = 0, 0
            for sim in tqdm(range(len(self.train_set)), ncols = 100):
                train_loader = DataLoader(self.train_set[sim], batch_size=self.batch_size, shuffle=True)

                for snaps in train_loader:
                    snaps = snaps.to(self.device)

                    # Get data
                    z_norm, z1_norm = self.norm(snaps.x, self.stats_z), self.norm(snaps.y, self.stats_z)
                    n = snaps.n
                    edge_index = snaps.edge_index

                    q_0_norm = self.norm(snaps.q_0, self.stats_q) if 'q_0' in snaps.keys else None
                    f_norm = self.norm(snaps.f, self.stats_q) if 'f' in snaps.keys else None
                    g = snaps.g if 'g' in snaps.keys else None
                    batch = snaps.batch if 'batch' in snaps.keys else None

                    # Add noise
                    noise = (self.noise_var)**0.5*torch.randn_like(z_norm[n[:,0]==1])
                    z_norm[n[:,0]==1] = z_norm[n[:,0]==1] + noise

                    # Net forward pass + Integration
                    L_net, M_net, dEdz_net, dSdz_net, _, _ = self.net(z_norm, n, edge_index, q_0=q_0_norm, f=f_norm, g=g, batch=batch)
                    dzdt_net, deg_E, deg_S = self.integrator(L_net, M_net, dEdz_net, dSdz_net)
                    dzdt = (z1_norm - z_norm)/self.dt

                    # Compute loss
                    loss_z = (((dzdt - dzdt_net))**2)[n[:,0]==1].mean()
                    loss_deg_E = (deg_E**2)[n[:,0]==1].mean()
                    loss_deg_S = (deg_S**2)[n[:,0]==1].mean()
                    loss = self.lambda_d*loss_z + (loss_deg_E + loss_deg_S)

                    loss_z_sum += loss_z.item()
                    loss_deg_E_sum += loss_deg_E.item()
                    loss_deg_S_sum += loss_deg_S.item()

                    # Backpropagation
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

            # Learning rate scheduler
            self.scheduler.step()

            # Train log 
            loss_z_train = loss_z_sum / len(train_loader) / len(self.train_set)
            loss_deg_E_train = loss_deg_E_sum / len(train_loader) / len(self.train_set)
            loss_deg_S_train = loss_deg_S_sum / len(train_loader) / len(self.train_set)
            train_log['epoch'].append(epoch+1)
            train_log['loss_z'].append(loss_z_train)
            train_log['loss_deg_E'].append(loss_deg_E_train)
            train_log['loss_deg_S'].append(loss_deg_S_train)

            # Validation set loop
            loss_z_sum = 0
            loss_deg_E_sum, loss_deg_S_sum = 0, 0
            for sim in range(len(self.val_set)):
                val_loader = DataLoader(self.val_set[sim], batch_size=self.batch_size)

                for snaps in val_loader:
                    snaps = snaps.to(self.device)

                    # Get data
                    z_norm, z1_norm = self.norm(snaps.x, self.stats_z), self.norm(snaps.y, self.stats_z)
                    edge_index = snaps.edge_index
                    n = snaps.n
                    
                    q_0_norm = self.norm(snaps.q_0, self.stats_q) if 'q_0' in snaps.keys else None
                    f_norm = self.norm(snaps.f, self.stats_q) if 'f' in snaps.keys else None
                    g = snaps.g if 'g' in snaps.keys else None
                    batch = snaps.batch if 'batch' in snaps.keys else None

                    # Net forward pass + Integration
                    L_net, M_net, dEdz_net, dSdz_net, _, _ = self.net(z_norm, n, edge_index, q_0=q_0_norm, f=f_norm, g=g, batch=batch)
                    dzdt_net, deg_E, deg_S = self.integrator(L_net, M_net, dEdz_net, dSdz_net)
                    dzdt = (z1_norm - z_norm)/self.dt

                    # Compute loss
                    loss_z = (((dzdt - dzdt_net))**2)[n[:,0]==1].mean()
                    loss_deg_E = (deg_E**2)[n[:,0]==1].mean()
                    loss_deg_S = (deg_S**2)[n[:,0]==1].mean()

                    loss_z_sum += loss_z.item()
                    loss_deg_E_sum += loss_deg_E.item()
                    loss_deg_S_sum += loss_deg_S.item()

            # Validation log
            loss_z_val = loss_z_sum / len(val_loader) / len(self.val_set)
            loss_deg_E_val = loss_deg_E_sum / len(val_loader) / len(self.val_set)
            loss_deg_S_val = loss_deg_S_sum / len(val_loader) / len(self.val_set)
            val_log['epoch'].append(epoch+1)
            val_log['loss_z'].append(loss_z_val)  
            val_log['loss_deg_E'].append(loss_deg_E_val)
            val_log['loss_deg_S'].append(loss_deg_S_val)

            # Print Loss
            print('Data Loss:    {:1.2e} (Train) / {:1.2e} (Val)'.format(loss_z_train, loss_z_val))
            print('Deg Loss (E): {:1.2e} (Train) / {:1.2e} (Val)'.format(loss_deg_E_train, loss_deg_E_val))
            print('Deg Loss (S): {:1.2e} (Train) / {:1.2e} (Val)\n'.format(loss_deg_S_train, loss_deg_S_val))

            epoch += 1

        print("[Training Finished]\n")

        # Save net parameters
        file_name = 'params_' + self.sys_name + '.pt'
        save_dir = os.path.join(self.output_dir, file_name)
        torch.save(self.net.state_dict(), save_dir)

        # Save logs
        save_log(self.args, train_log, 'train')
        save_log(self.args, val_log, 'val')


    def test_model(self):
        
        print("[Train Set Evaluation]")
        train_error = self.compute_error(self.train_set)
        print_error(train_error)
        print("[Train Evaluation Finished]\n")

        print("[Test Set Evaluation]")
        test_error = self.compute_error(self.test_set)
        print_error(test_error)
        print("[Test Evaluation Finished]\n")


    # Plot a single simulation
    def plot_sim(self, sim=0):

        data_list = self.test_set[sim]

        if self.sys_name == 'beam': 
            print("[Plotting]")
            z_net, z_gt, _, _ = self.integrate_sim(data_list)
            plot_3D(z_net, z_gt, data_list, self.output_dir)
            print("[Plot Saved]\n")

        elif self.sys_name == 'cylinder': 
            print("[Plotting]")
            z_net, z_gt, _, _ = self.integrate_sim(data_list)
            plot_2D(z_net, z_gt, data_list, self.output_dir)
            print("[Plot Saved]\n")


    # Compute error of all the dataset
    def compute_error(self, dataset):

        if self.sys_name == 'couette': error = dict({'q':[], 'v':[], 'e':[], 'tau':[]})
        elif self.sys_name == 'beam': error = dict({'q':[], 'v':[], 'sigma':[]}) 
        elif self.sys_name == 'cylinder': error = dict({'v':[], 'P':[]})

        for data_list in dataset:
            # Compute Simulations
            z_net, z_gt, _, _ = self.integrate_sim(data_list)
            
            # Compute error
            e = z_net[1:].numpy()-z_gt[1:].numpy()
            gt = z_gt[1:].numpy()

            if self.sys_name == 'couette':
                # Position + Velocity + Energy + Conformation Tensor
                L2_q = ((e[:,:,[0,1]]**2).sum((1,2)) / (gt[:,:,[0,1]]**2).sum((1,2)))**0.5
                L2_v = ((e[:,:,2]**2).sum(1) / (gt[:,:,2]**2).sum(1))**0.5
                L2_e = ((e[:,:,3]**2).sum(1) / (gt[:,:,3]**2).sum(1))**0.5
                L2_tau = ((e[:,:,4]**2).sum(1) / (gt[:,:,4]**2).sum(1))**0.5 

                error['q'].extend(list(L2_q))
                error['v'].extend(list(L2_v))
                error['e'].extend(list(L2_e))
                error['tau'].extend(list(L2_tau))   

            elif self.sys_name == 'beam':
                # Position + Velocity + Stress Tensor
                L2_q = ((e[:,:,0:3]**2).sum((1,2)) / (gt[:,:,0:3]**2).sum((1,2)))**0.5
                L2_v = ((e[:,:,3:6]**2).sum((1,2)) / (gt[:,:,3:6]**2).sum((1,2)))**0.5
                L2_sigma = ((e[:,:,6:]**2).sum((1,2)) / (gt[:,:,6:]**2).sum((1,2)))**0.5 

                error['q'].extend(list(L2_q))
                error['v'].extend(list(L2_v))
                error['sigma'].extend(list(L2_sigma))

            elif self.sys_name == 'cylinder':
                # Velocity + Pressure
                L2_v = ((e[:,:,[0,1]]**2).sum((1,2)) / (gt[:,:,[0,1]]**2).sum((1,2)))**0.5
                L2_P = ((e[:,:,2]**2).sum(1) / (gt[:,:,2]**2).sum(1))**0.5             

                error['v'].extend(list(L2_v))
                error['P'].extend(list(L2_P))

        return error  


    # Integrate a single simulation
    def integrate_sim(self, data_list, full_rollout=True):

        N_nodes = data_list[0].x.size(0)
        dim_z = self.dims['z']

        # Preallocation
        z_net = torch.zeros(len(data_list)+1, N_nodes, dim_z)
        z_gt = torch.zeros(len(data_list)+1, N_nodes, dim_z)
        E = torch.zeros(len(data_list), N_nodes, 1)
        S = torch.zeros(len(data_list), N_nodes, 1)

        # Initial conditions
        z_net[0] = data_list[0].x
        z_gt[0] = data_list[0].x

        # Rollout loop
        z = data_list[0].x.to(self.device)
        z_norm = self.norm(z, self.stats_z)
        loader = DataLoader(data_list)
        for t, snap in enumerate(loader):
            snap = snap.to(self.device)

            # Get data   
            edge_index = snap.edge_index
            n = snap.n

            q_0_norm = self.norm(snap.q_0, self.stats_q) if 'q_0' in snap.keys else None
            f_norm = self.norm(snap.f, self.stats_q) if 'f' in snap.keys else None
            g = snap.g if 'g' in snap.keys else None
            batch = snap.batch if 'batch' in snap.keys else None

            # Net forward pass + Integration
            L_net, M_net, dEdz_net, dSdz_net, E_net, S_net = self.net(z_norm, n, edge_index, q_0=q_0_norm, f=f_norm, g=g, batch=batch)
            dzdt_net, _, _ = self.integrator(L_net, M_net, dEdz_net, dSdz_net)
            z1_net = z_norm + self.dt*dzdt_net

            # Boundary Conditions
            for bc in range(n.size(1)-1):
                z1_net[n[:,bc+1]==1] = self.norm(snap.y[n[:,bc+1]==1], self.stats_z)

            # Save results
            z_net[t+1] = self.denorm(z1_net.detach(), self.stats_z)
            z_gt[t+1] = snap.y
            E[t] = E_net.detach()
            S[t] = S_net.detach()

            # Update
            z_norm = z1_net.detach() if full_rollout else self.norm(snap.y, self.stats_z)

        return z_net, z_gt, E.sum(dim=1), S.sum(dim=1)


    # Normalization function
    def norm(self, z, stats):
        return (z - stats['mean']) / stats['std']


    # Denormalization function
    def denorm(self, z, stats):
        return z * stats['std'] + stats['mean']


    # Forward-Euler Integrator
    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.bmm(L,dEdz) + torch.bmm(M,dSdz)
        deg_E = torch.bmm(M,dEdz)
        deg_S = torch.bmm(L,dSdz)

        return dzdt[:,:,0], deg_E[:,:,0], deg_S[:,:,0]


if __name__ == '__main__':
    pass
