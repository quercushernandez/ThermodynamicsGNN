"""plot.py"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_3D(z_net, z_gt, data_list, output_dir, var=6):
    T = z_net.size(0)
    n = data_list[0].n

    # Plot initialization
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

    # Adjust ranges
    X, Y, Z = z_gt[:,:,0].numpy(), z_gt[:,:,1].numpy(), z_gt[:,:,2].numpy()
    z_min, z_max = z_gt[:,:,var].min(), z_gt[:,:,var].max()
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[0,:,0], z_net[0,:,1], z_net[0,:,2]
    q1_gt, q2_gt, q3_gt = z_gt[0,:,0], z_gt[0,:,1], z_gt[0,:,2]
    var_net, var_gt = z_net[0,:,var], z_gt[0,:,var]
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [yb], [zb], 'w')
        ax2.plot([xb], [yb], [zb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net[n[:,0]==1], q2_net[n[:,0]==1], q3_net[n[:,0]==1], c=var_net[n[:,0]==1], vmax=z_max, vmin=z_min)
    ax1.scatter(q1_net[n[:,1]==1], q2_net[n[:,1]==1], q3_net[n[:,1]==1], color='k')
    s2 = ax2.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], c=var_gt[n[:,0]==1], vmax=z_max, vmin=z_min)
    ax2.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')    
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom')
    fig.colorbar(s2, ax=ax2, location='bottom')
    
    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
            ax2.plot([xb], [yb], [zb], 'w')
        # Scatter points
        q1_net, q2_net, q3_net = z_net[snap,:,0], z_net[snap,:,1], z_net[snap,:,2]
        q1_gt, q2_gt, q3_gt = z_gt[snap,:,0], z_gt[snap,:,1], z_gt[snap,:,2]
        var_net, var_gt = z_net[snap,:,var], z_gt[snap,:,var]
        ax1.scatter(q1_net[n[:,0]==1], q2_net[n[:,0]==1], q3_net[n[:,0]==1], c=var_net[n[:,0]==1], vmax=z_max, vmin=z_min)
        ax1.scatter(q1_net[n[:,1]==1], q2_net[n[:,1]==1], q3_net[n[:,1]==1], color='k')
        ax2.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], c=var_gt[n[:,0]==1], vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')

        return fig,
    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20) 

    # Save as gif 
    save_dir = os.path.join(output_dir, 'beam.gif')
    anim.save(save_dir, writer=writergif)


def plot_2D(z_net, z_gt, data_list, output_dir, var=0):
    T = z_net.size(0)
    res = 100
    q_0 = data_list[0].q_0
    n = data_list[0].n

    # Plot initialization
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')

    # Adjust ranges
    X, Y = q_0[:,0].cpu().numpy(), q_0[:,1].cpu().numpy()
    z_min, z_max = z_gt[:,:,var].min(), z_gt[:,:,var].max()
    levels = np.linspace(z_min, z_max, 10, endpoint = True)
    ax1.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax2.axis([X.min(), X.max(), Y.min(), Y.max()])

    # Interpolate data to grid
    zi_net = np.zeros([T, res, res])
    zi_gt = np.zeros([T, res, res])
    xi = np.linspace(X.min(), X.max(), res)
    yi = np.linspace(Y.min(), Y.max(), res)
    for t in range(T):
        Z_gt = z_gt[t,:,var].numpy()
        Z_net = z_net[t,:,var].numpy()
        zi_net[t] = griddata((X, Y), Z_net, (xi[None,:], yi[:,None]), method='cubic')
        zi_gt[t] = griddata((X, Y), Z_gt, (xi[None,:], yi[:,None]), method='cubic')

    # Initial snapshot
    c1 = ax1.contourf(xi, yi, zi_net[0], levels=levels, vmax=z_max, vmin=z_min)
    c2 = ax2.contourf(xi, yi, zi_gt[0], levels=levels, vmax=z_max, vmin=z_min)
    # Plot Boundaries
    ax1.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
    ax2.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    # Colorbar
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(c1, ax=ax1, cax=cax1, ticks=np.linspace(z_min, z_max, 5), format='%.2f')
    fig.colorbar(c2, ax=ax2, cax=cax2, ticks=np.linspace(z_min, z_max, 5), format='%.2f')

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        # Boundaries
        ax1.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
        ax2.plot(q_0[n[:,3]==1,0], q_0[n[:,3]==1,1],'k.')
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        # Contour
        ax1.contourf(xi, yi, zi_net[snap], levels=levels, vmax=z_max, vmin=z_min)
        ax2.contourf(xi, yi, zi_gt[snap], levels=levels, vmax=z_max, vmin=z_min)
        return fig,
    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=30) 

    # Save as gif
    save_dir = os.path.join(output_dir, 'cylinder.gif')
    anim.save(save_dir, writer=writergif)


