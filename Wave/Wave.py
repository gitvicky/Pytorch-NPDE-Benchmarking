#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:37:41 2020

@author: vgopakum


PDE: u_t - 1.0*(u_xx + u_yy)
IC: u(0, x) = e^(-40*((x-0.4)**2 + y**2))
BC: Dirichlet = 0
Domain: t ∈ [0, 1],  x ∈ [-1, 1],  y ∈ [-1, 1]

Initial Velocity: u_t(0,x,y)=0
"""

import time 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import scipy.io

from pyDOE import lhs
import torch 
import torch.nn as nn 

default_device = "cuda" if torch.cuda.is_available() else "cpu"
device_1 = torch.device("cuda:0")

# device_2 = torch.device("cuda:1")
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

# %%
import wandb 

bench_name = 'Regular Net'

configuration={"Nummber of Layers": 4,
                "Number of Neurons": 64,
                "Activation": 'tanh',
                "Optimizer": 'Adam',
                "Learning rate": 0.001,
                "Epochs": 5000,
                "Batch size": None,
                "Quasi-newtonian": 'LBFGS',
                "QN epochs": 50,
                "N_domain": 20000,
                "N_initial": 500,
                "N_boundary": 2000}



wandb.init(project='Pytorch NPDE Benchmark - Wave', name=bench_name, 
            notes='Unnormalised Inputs - Initial, Boundary and Domain Sampled.',
            config=configuration)

# %%

def min_max_norm(x, lb, ub):
    return (x - lb)/(ub - lb) 

def standardisation(x):
    return (x - x.mean())/ np.sqrt(x.var()**2)

# %%

def torch_tensor_grad(x, device=default_device):
    return torch.autograd.Variable(torch.tensor(x, dtype=torch.float64).float(), requires_grad=True).to(device)

def torch_tensor_nograd(x, device=default_device):
    return torch.tensor(x, dtype=torch.float64).float().to(device)

# %%

N_i = configuration['N_initial']
N_b = configuration['N_boundary']
N_f = configuration['N_domain']

lb, ub = np.asarray([0.0, -1.0, -1.0]), np.asarray([1.0, 1.0, 1.0])

#Unsupervised approach with no training data 
def uniform_sampler(lb, ub, dims, N):
    return  np.asarray(lb) + (np.asarray(ub)-np.asarray(lb))*lhs(dims, N)

IC = lambda x, y: np.exp(-40*((x-0.4)**2 + y**2))


X_i = uniform_sampler([lb[0], lb[1], lb[2]], [lb[0], ub[1], ub[2]], 3, N_i)
X_lower = uniform_sampler([lb[0], lb[1], lb[2]], [ub[0], ub[1], lb[2]], 3, N_b)
X_upper = uniform_sampler([lb[0], lb[1], ub[2]], [ub[0], ub[1], ub[2]], 3, N_b)
X_left = uniform_sampler([lb[0], lb[1], lb[2]], [ub[0], lb[1], ub[2]], 3, N_b)
X_right = uniform_sampler([lb[0], ub[1], lb[2]], [ub[0], ub[1], ub[2]], 3, N_b)
X_f = uniform_sampler(lb, ub, 3, N_f)
u_i = IC(X_i[:,1:2], X_i[:,2:3])

X_i_torch = torch_tensor_grad(X_i, device_1)
X_lower_torch = torch_tensor_grad(X_lower, device_1)
X_upper_torch = torch_tensor_grad(X_upper, device_1)
X_left_torch = torch_tensor_grad(X_left, device_1)
X_right_torch = torch_tensor_grad(X_right, device_1)
X_f_torch = torch_tensor_grad(X_f, device_1)
u_i_torch = torch_tensor_nograd(u_i, device_1)

# %%
class Net(nn.Module):
    def __init__(self, in_features = 2, out_features=1, num_neurons = 64, activation=torch.nn.Tanh):
        super(Net, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        
        self.act_func = activation()
        
        self.layer1 = nn.Linear(self.in_features, self.num_neurons)
        self.layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.layer3 = nn.Linear(self.num_neurons, self.num_neurons)
        self.layer4 = nn.Linear(self.num_neurons, self.num_neurons)


        self.layer_output = nn.Linear(self.num_neurons, self.out_features)
        
        self.net = [self.layer1, self.layer2, self.layer3, self.layer4]
        
    def forward(self, x):
        x_temp = x

        for dense in self.net:
            x_temp = self.act_func(dense(x_temp))
        
        x_temp = self.layer_output(x_temp)
        return x_temp
    
    
    
    
class Resnet(nn.Module):
    def __init__(self, in_features = 2, out_features=1, num_neurons = 64, activation=torch.nn.Tanh):
        super(Resnet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        
        self.act_func = activation()
        
        self.block1_layer1 = nn.Linear(self.in_features, self.num_neurons)
        # self.block1_bn1 = nn.BatchNorm1d(self.num_neurons)
        self.block1_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        # self.block1_bn2 = nn.BatchNorm1d(self.num_neurons)
        self.block1 = [self.block1_layer1, self.block1_layer2]
        
        self.block2_layer1 = nn.Linear(self.in_features + self.num_neurons, self.num_neurons)
        # self.block2_bn1 = nn.BatchNorm1d(self.num_neurons)
        self.block2_layer2 = nn.Linear(self.num_neurons, self.num_neurons)
        # self.block2_bn2 = nn.BatchNorm1d(self.num_neurons)
        self.block2 = [self.block2_layer1, self.block2_layer2]
        
        self.layer_after_block = nn.Linear(self.num_neurons + self.in_features, self.num_neurons)
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)
        
        
    def forward(self, x):
        
        x_temp = x
        
        for dense in self.block1:
            x_temp = self.act_func(dense(x_temp))
        
        x_temp = torch.cat([x_temp, x], dim=-1)
        
        for dense in self.block2:
            x_temp = self.act_func(dense(x_temp))
        
        x_temp = torch.cat([x_temp, x], dim=-1)
        x_temp = self.act_func(self.layer_after_block(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp
    
    
# %%
class CustomLinearLayer(nn.Module):
    def __init__(self,in_size, out_size):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_size, out_size))
        self.bias = nn.Parameter(torch.zeros(out_size))
        
    def forward(self, x):
        return x.mm(self.weights) + self.bias
    
    
class CustomNet(nn.Module):
    def __init__(self, in_features = 2, out_features=1, num_neurons = 64, activation=torch.nn.Tanh):
        super(CustomNet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons
        
        self.act_func = activation()
        
        self.layer1 = CustomLinearLayer(self.in_features, self.num_neurons)
        self.layer2 = CustomLinearLayer(self.num_neurons, self.num_neurons)
        self.layer3 = CustomLinearLayer(self.num_neurons, self.num_neurons)
        self.layer4 = CustomLinearLayer(self.num_neurons, self.num_neurons)


        self.layer_output = CustomLinearLayer(self.num_neurons, self.out_features)
        
        self.net = [self.layer1, self.layer2, self.layer3, self.layer4]
        
    def forward(self, x):
        x_temp = x

        for dense in self.net:
            x_temp = self.act_func(dense(x_temp))
        
        x_temp = self.layer_output(x_temp)
        return x_temp
    
    
    
# %%

npde_net = Net()
npde_net = npde_net.to(default_device)
wandb.watch(npde_net, log='all')

deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
 

# %%

def recon_loss(x, y):
    f = (npde_net(x) - y).pow(2).mean()
    return f


def bc_loss(X_lower, X_upper, X_left, X_right):

    u_lower = npde_net(X_lower)
    u_upper = npde_net(X_upper)
    u_left = npde_net(X_left)
    u_right = npde_net(X_right)

    return (u_lower + u_upper + u_left + u_right).pow(2).mean() 
    
def npde_loss(X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    y = X[:, 2:3]
    u = npde_net(torch.cat([t, x],1))

    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)
    u_y = deriv(u, y)
    u_yy = deriv(u_y, y)
    u_t = deriv(u, t)
    
    f = u_t - (u_xx + u_yy)
    return f.pow(2).mean()

# %%

learning_rate = configuration['Learning rate']
optimizer = torch.optim.Adam(npde_net.parameters(), lr=learning_rate)


# %%

start_time = time.time()


epochs = configuration['Epochs']
for it in tqdm(range(epochs)):
    optimizer.zero_grad()
    
    initial_loss = recon_loss(X_i_torch, u_i_torch) 
    boundary_loss = bc_loss(X_lower_torch, X_upper_torch, X_left_torch, X_right_torch)
    domain_loss = npde_loss(X_f_torch)

    loss = domain_loss + initial_loss + boundary_loss
    
    wandb.log({'Initial Loss': initial_loss, 
               'Boundary Loss': boundary_loss,
               'Domain Loss': domain_loss,
               'Total Loss ': loss})

    
    loss.backward()
    optimizer.step()
    
    print('Total.  It: %d, Loss: %.3e' % (it, loss.item()))

SGD_time = time.time() - start_time 
wandb.run.summary['SGD Time'] = SGD_time

# %%
    
start_time = time.time() 
optimizer = torch.optim.LBFGS(npde_net.parameters(),
                              lr=1.0, 
                              max_iter=5000, 
                              max_eval=None, 
                              tolerance_grad=1e-07,
                              tolerance_change=1e-09)

    
def closure():    
    optimizer.zero_grad()
     
    domain_loss = npde_loss(X_f_torch)
    initial_loss = recon_loss(X_i_torch, u_i_torch) 
    boundary_loss = bc_loss(X_lower_torch, X_upper_torch, X_left_torch, X_right_torch)
  
    
    loss = domain_loss + initial_loss + boundary_loss

    wandb.log({"QN Loss": loss})
    print('QN.  It: %d, Loss: %.3e' % (ii, loss.item()))
    loss.backward()
    return loss
    

qn_epochs = configuration['QN epochs']
for ii in tqdm(range(qn_epochs)):
    optimizer.step(closure)

QN_time = time.time() - start_time 
wandb.run.summary['QN Time'] = QN_time

wandb.run.summary['Total Time'] = SGD_time + QN_time

# %%

data_loc = os.path.abspath('.') + '/Data/'
data =np.load(data_loc +'Wave.npz')

t = data['t']
x = data['x']
y = data['y']

xx, yy = np.meshgrid(x,y)
tt = np.repeat(t, len(x)*len(y))

X_star = np.column_stack((tt, np.tile(xx.flatten(), len(t)), 
                     np.tile(yy.flatten(), len(t))))
Exact = data['u']

# %%

with torch.no_grad():
    u_pred = npde_net(torch_tensor_grad(X_star)).cpu().detach().numpy()
    
u_pred = np.reshape(u_pred, np.shape(Exact))
u_actual = Exact

# %%

from matplotlib import cm
def animation(wave_name, u_field):

    
    def update_plot(frame_number, u_field, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(xx, yy, u_field[frame_number], cmap=cm.coolwarm, linewidth=2, antialiased=False)
        
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    plot = [ax.plot_surface(xx, yy, u_field[0], cmap=cm.coolwarm, linewidth=2, antialiased=False)]
    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(-1.0, 1.0)
    ax.set_zlim3d(-0.15, 1)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U")
    ax.set_title(wave_name)
    
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    #plt.tight_layout()
    ax.view_init(elev=30., azim=-110)
    
    
    fps = 50 # frame per sec
    frn = len(u_field) # frame number of the animation
    
    ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(u_field, plot), interval=1000/fps)
    
    
    ani.save(wave_name+'.gif',writer='imagemagick',fps=fps)
    
    fn_gif = wave_name + '.gif'
    wandb.log({wave_name + '' + bench_name: wandb.Video(fn_gif, fps=10, format="gif")})
    
animation('Numerical', u_actual)
animation('Neural Network', u_pred)

