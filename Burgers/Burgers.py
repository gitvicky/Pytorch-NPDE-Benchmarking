#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:21:09 2020


@author: vgopakum

Neural PDE test for Burgers using Pytorch 

PDE: u_t + u*u_x - 0.1*u_xx
IC: u(0, x) = -sin(pi.x/8)
BC: Periodic 
Domain: t ∈ [0,10],  x ∈ [-8,8]

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
                "N_domain": 5000,
                "N_initial": 100,
                "N_boundary": 1000}



wandb.init(project='Pytorch NPDE Benchmark - Burgers', name=bench_name, 
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

lb, ub = np.asarray([0.0, -8.0]), np.asarray([10.0, 8.0])

#Unsupervised approach with no training data 
def uniform_sampler(lb, ub, dims, N):
    return  np.asarray(lb) + (np.asarray(ub)-np.asarray(lb))*lhs(dims, N)

IC = lambda x: -np.sin(np.pi*x/8)


X_i = uniform_sampler([lb[0], lb[1]], [lb[0], ub[1]], 2, N_i)
X_lb = uniform_sampler([lb[0], lb[1]], [ub[0], lb[1]], 2, N_b)
X_ub = uniform_sampler([lb[0], ub[1]], [ub[0], ub[1]], 2, N_b)
X_f = uniform_sampler(lb, ub, 2, N_f)
u_i = IC(X_i[:,1:2])

X_i_torch = torch_tensor_grad(X_i, device_1)
X_lb_torch = torch_tensor_grad(X_lb, device_1)
X_ub_torch = torch_tensor_grad(X_ub, device_1)
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


def bc_loss(X_lb, X_ub):
    
    t_lb = X_lb[:,0:1]
    t_ub = X_ub[:,0:1]
    x_lb = X_lb[:,1:2]
    x_ub = X_ub[:,1:2]

    u_lb = npde_net(torch.cat([t_lb, x_lb], 1))
    u_ub = npde_net(torch.cat([t_ub, x_ub], 1))
    
    u_x_lb = deriv(u_lb, x_lb)
    u_x_ub = deriv(u_ub, x_ub)


    return (u_lb - u_ub).pow(2).mean() + (u_x_lb - u_x_ub).pow(2).mean()
    
def npde_loss(X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    u = npde_net(torch.cat([t, x],1))

    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)
    u_t = deriv(u, t)
    
    f = u_t + u*u_x - 0.1*u_xx 
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
    boundary_loss = bc_loss(X_lb_torch, X_ub_torch)
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
    boundary_loss = bc_loss(X_lb_torch, X_ub_torch)
  
    
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
data = scipy.io.loadmat(data_loc +'burgers_sine.mat')



t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T


X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) 
u_star = Exact.flatten()[:,None]    

# %%

with torch.no_grad():
    u_pred = npde_net(torch_tensor_grad(X_star)).cpu().detach().numpy()
    
u_pred = np.reshape(u_pred, np.shape(Exact))
u_actual = Exact

# %%

from matplotlib import animation

def animate(u_pred, name):
    
    def update_plot(ii, x, t, u_actual, u_pred, line1, line2, ax):
        line1.set_data(x, u_actual[ii])
        line2.set_data(x, u_pred[ii])
        mse = np.round(np.mean((u_actual[ii]-u_pred[ii])**2), 3)
        time_text.set_text("Time : " + str(t[ii]))
        loss_text.set_text("Loss : " + str(mse))
    
        return [line1,line2, time_text, time_text, loss_text]
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    
    line1, = ax.plot([], [], lw=1.5, label="Actual")
    line2, = ax.plot([], [], lw=1.5, label="Prediction")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    loss_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    
    ax.set_xlim(( lb[1], ub[1]))
    ax.set_ylim((-3, 3))
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_title('Actual vs Prediction')
    
    
    anim = animation.FuncAnimation(fig, update_plot, fargs=(x, t, u_actual, u_pred, line1, line2, ax),
                                   frames=len(t), interval=40, blit=True)
    
    anim
    
    fn = name
    anim.save(fn+'.gif',writer='imagemagick',fps=10)
    
    fn_gif = fn + '.gif'
    wandb.log({fn: wandb.Video(fn_gif, fps=10, format="gif")})

animate(u_pred, bench_name)