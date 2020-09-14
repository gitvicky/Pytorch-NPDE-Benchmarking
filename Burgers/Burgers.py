#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 19:16:24 2020

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
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

# %%

# %%
import wandb 

configuration={"NUM_LAYERS": 4,
                "NUM_NEURONS": 64,
                "ACTIVATION": 'tanh',
                "OPTIMIZER": 'Adam',
                "LR": 0.001,
                "EPOCHS": 5000,
                "BATCH SIZE": None,
                "QUASI-NEWTONIAN": 'LBFGS',
                "QN epochs": 200,
                "N_domain": 5000,
                "N_initial": 100,
                "N_boundary": 1000,
                "Retrain Points - N_f": None,
                "Retrain Epochs"  :None }


wandb.init(project='Pytorch-PDE_expts', name='NPDE Test - Burgers', 
            notes='NPDE fit for Burgers Equation with data gathered across space time. Trying out deriv as a Lambda function for both PDE and BC',
            config=configuration)
# %%

data_loc = os.path.abspath('.') + '/Data/'
data = scipy.io.loadmat(data_loc +'burgers_sine.mat')

# %%
N_f = 5000
N_i = 100
N_b = 100


t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T


X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) 
u_star = Exact.flatten()[:,None]              

# Domain bounds
lb = X_star.min(0) 
ub = X_star.max(0)
    
X_i = np.hstack((T[0:1,:].T, X[0:1,:].T))
u_i = Exact[0:1,:].T

X_lb = np.hstack((T[:,0:1], X[:,0:1])) 
u_lb = Exact[:,0:1] 
X_ub = np.hstack((T[:,-1:], X[:,-1:])) 
u_ub = Exact[:,-1:] 

u_lb = np.zeros((len(u_lb),1))
u_ub = np.zeros((len(u_ub),1))  

X_b = np.vstack((X_lb, X_ub))
u_b = np.vstack((u_lb, u_ub))

X_f = lb + (ub-lb)*lhs(2, N_f)

idx = np.random.choice(X_i.shape[0], N_i, replace=False)
X_i = X_i[idx, :]
u_i = u_i[idx,:]

idx = np.random.choice(X_b.shape[0], N_b, replace=False)
X_b = X_b[idx, :] 
u_b = u_b[idx,:]

def min_max_norm(x):
    return 2*(x - lb)/(ub - lb) - 1


# %%

def torch_tensor_grad(x):
    return torch.autograd.Variable(torch.tensor(x, dtype=torch.float64, device=device).float(), requires_grad=True)

def torch_tensor_nograd(x):
    return torch.tensor(x, dtype=torch.float64, device=device).float()

X_i_torch = torch_tensor_grad(X_i)
X_b_torch = torch_tensor_grad(X_b)
X_f_torch = torch_tensor_grad(X_f)
u_i_torch = torch_tensor_nograd(u_i)
u_b_torch = torch_tensor_nograd(u_b)


training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}


# %%
class resnet(nn.Module):
    def __init__(self, in_features = 2, out_features=1, num_neurons = 64, activation=nn.Tanh):
        super(resnet, self).__init__()
        
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
        # x_temp = self.act_func(self.layer1(x_temp))
        # x_temp = self.act_func(self.layer2(x_temp))
        # x_temp = self.act_func(self.layer3(x_temp))
        # x_temp = self.act_func(self.layer4(x_temp))

        for dense in self.net:
            x_temp = self.act_func(dense(x_temp))
         
        x_temp = self.layer_output(x_temp)
        return x_temp
    


npde_net = resnet().to(device)
wandb.watch(npde_net, log='all')

deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]

# def npde_loss(x):
#     u = npde_net(x)
#     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True, only_inputs=True)[0]
#     u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, allow_unused=True, only_inputs = True)[0]
#     f = u_x[:, 0:1] + u*u_x[:, 1:2] - 0.1*u_xx[:, 1:2] 
#     return f.pow(2).mean()

def npde_loss(X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    u = npde_net(torch.cat([t, x],1))
    # u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True, only_inputs=True)[0]
    # u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True, only_inputs=True)[0]
    # u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, only_inputs = True)[0]
    u_x = deriv(u, x)
    u_xx = deriv(u_x, x)
    u_t = deriv(u, t)
    f = u_t + u*u_x - 0.1*u_xx 
    return f.pow(2).mean()


def recon_loss(x, y):
    f = (npde_net(x) - y).pow(2).mean()
    return f


learning_rate = 1e-2
optimizer = torch.optim.Adam(npde_net.parameters(), lr=learning_rate)


# %%

# initial_train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_tensor_grad(X_i), torch_tensor_nograd(IC(X_i[:,1]))), batch_size=512, shuffle=True)
# boundary_train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_tensor_grad(X_lb), torch_tensor_grad(X_ub)), batch_size=128, shuffle=True)
# domain_train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_tensor_grad(X_f)), batch_size=512, shuffle=True)

# %%

'''
start_time = time.time()

epochs = configuration['EPOCHS']
for it in tqdm(range(epochs)):
    optimizer.zero_grad()
    
    domain_loss = npde_loss(X_f_torch)
    initial_loss = recon_loss(X_i_torch, u_i_torch) 
    boundary_loss = recon_loss(X_b_torch, u_b_torch)
    
    print('IC.  It: %d, Loss: %.3e' % (it, initial_loss.item()))
    
    loss = domain_loss + initial_loss + boundary_loss
    
    loss.backward()
    optimizer.step()

    print('GD.  It: %d, Loss: %.3e' % 
                  (it, loss.item()))



optimizer = torch.optim.LBFGS(npde_net.parameters(),
                              lr=1.0, 
                              max_iter=20, 
                              max_eval=None, 
                              tolerance_grad=1e-07,
                              tolerance_change=1e-09)

def closure():    
    optimizer.zero_grad()
     
    domain_loss = npde_loss(X_f_torch)
    initial_loss = recon_loss(X_i_torch, u_i_torch) 
    boundary_loss = recon_loss(X_b_torch, u_b_torch)
    
  
    
    loss = domain_loss + initial_loss + boundary_loss


    print('QN.  It: %d, Loss: %.3e' % (it, loss.item()))
    loss.backward()
    return loss
    
for ii in tqdm(range(configuration['QN epochs'])):
    optimizer.step(closure)

    
npde_time = time.time() - start_time




with torch.no_grad():
    outputs_pred = npde_net(torch_tensor_grad(X_star)).detach().numpy()

#%%

def evolution_plot(u_actual, u_pred):
    """
    Plots frame by frame evolution of the neural network solution in the 
    1D space along with the baseline solution
    Parameters
    ----------
    u_actual : 2D NUMPY ARRAY
        True solution 
    u_pred : 2D NUMPY ARRAY
        Neural Network Prediction
    Returns
    -------
    None.
    """
    
    actual_col = '#302387'
    nn_col = '#DF0054'
    plt.figure()
    plt.plot(0, 0, c = actual_col, label='Actual')
    plt.plot(0, 0, c = nn_col, label='NN', alpha = 0.5)
    plt.legend()
    for ii in range(len(u_actual)):
        plt.plot(u_actual[ii], c= actual_col, label = "Actual")
        plt.plot(u_pred[ii], c= nn_col, label = "NN")
        plt.legend()
        plt.pause(0.0001)
        plt.clf()

# u_pred = np.reshape(outputs_pred, np.shape(Exact))
# u_actual = Exact

# evolution_plot(u_actual, u_pred)
'''



# %% 

#Unsupervised approach with no training data 
def uniform_sampler(lb, ub, dims, N):
    return  np.asarray(lb) + (np.asarray(ub)-np.asarray(lb))*lhs(dims, N)

IC = lambda x: -np.sin(np.pi*x/8)

def BC(X_lb, X_ub):
    
    t_lb = X_lb[:,0:1]
    t_ub = X_ub[:,0:1]
    x_lb = X_lb[:,1:2]
    x_ub = X_ub[:,1:2]

    u_lb = npde_net(torch.cat([t_lb, x_lb], 1))
    u_ub = npde_net(torch.cat([t_ub, x_ub], 1))
    
    u_x_lb = deriv(u_lb, x_lb)
    u_x_ub = deriv(u_ub, x_ub)
    
    # u_x_lb = torch.autograd.grad(u_lb, x_lb, grad_outputs=torch.ones_like(u_lb), create_graph=True, allow_unused=True, only_inputs=True)[0]
    # u_x_ub = torch.autograd.grad(u_ub, x_ub, grad_outputs=torch.ones_like(u_ub), create_graph=True, allow_unused=True, only_inputs=True)[0]
    
    return (u_lb - u_ub).pow(2).mean() + (u_x_lb - u_x_ub).pow(2).mean()
    

N_i = configuration['N_initial']
N_b = configuration['N_boundary']
N_f = configuration['N_domain']

# X_i = uniform_sampler([0, -8], [0, 8], 2, 100)
X_lb = uniform_sampler([0, -8], [10, -8], 2, N_b)
X_ub = uniform_sampler([0, 8], [10, 8], 2, N_b)
# X_b = np.vstack((X_lb, X_ub))
X_f = uniform_sampler([0, -8], [10, 8], 2, N_f)
# u_i = IC(X_i[:,1])

X_i_torch = torch_tensor_grad(X_i)
X_lb_torch = torch_tensor_grad(X_lb)
X_ub_torch = torch_tensor_grad(X_ub)
X_f_torch = torch_tensor_grad(X_f)
u_i_torch = torch_tensor_nograd(u_i)
# u_b_torch = torch_tensor_nograd(u_b)

# %%

# initial_train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_tensor_grad(X_i), torch_tensor_nograd(IC(X_i[:,1]))), batch_size=512, shuffle=True)
# boundary_train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_tensor_grad(X_lb), torch_tensor_grad(X_ub)), batch_size=128, shuffle=True)
# domain_train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_tensor_grad(X_f)), batch_size=512, shuffle=True)

# %%
npde_net = resnet().to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(npde_net.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, pct_start=0.1, div_factor=10, final_div_factor=1, total_steps=epochs)

start_time = time.time()


epochs = configuration['EPOCHS']
for it in tqdm(range(epochs)):
    optimizer.zero_grad()
    
    domain_loss = npde_loss(X_f_torch)
    # domain_loss.backward(retain_graph=True)
    wandb.log({'Domain Loss': domain_loss})
    # print('Domain.  It: %d, Loss: %.3e' % (it, domain_loss.item()))

    initial_loss = recon_loss(X_i_torch, u_i_torch) 
    # initial_loss.backward(retain_graph=True)
    wandb.log({'Initial Loss': initial_loss})
    # print('IC.  It: %d, Loss: %.3e' % (it, initial_loss.item()))
    
    
    boundary_loss = BC(X_lb_torch, X_ub_torch)
    # boundary_loss.backward()
    wandb.log({'Boundary Loss': boundary_loss})
    # print('BC.  It: %d, Loss: %.3e' % (it, boundary_loss.item()))

    
    loss = domain_loss + initial_loss + boundary_loss
    wandb.log({'Total Loss': loss})
    
    wandb.log({'Domain Loss': domain_loss, 
               'Boundary Loss': boundary_loss,
               'Initial Loss': initial_loss,
               'Total Loss ': loss})

    
    loss.backward()
    optimizer.step()
    
    print('Total.  It: %d, Loss: %.3e' % (it, loss.item()))
    
    
    

optimizer = torch.optim.LBFGS(npde_net.parameters(),
                              lr=1.0, 
                              max_iter=20, 
                              max_eval=None, 
                              tolerance_grad=1e-07,
                              tolerance_change=1e-09)

def closure():    
    optimizer.zero_grad()
     
    domain_loss = npde_loss(X_f_torch)
    initial_loss = recon_loss(X_i_torch, u_i_torch) 
    boundary_loss = BC(X_lb_torch, X_ub_torch)
    
  
    
    loss = domain_loss + initial_loss + boundary_loss

    wandb.log({"QN Loss": loss})
    print('QN.  It: %d, Loss: %.3e' % (ii, loss.item()))
    loss.backward()
    return loss
    

for ii in tqdm(range(200)):
    optimizer.step(closure)

total_time = time.time() - start_time 

wandb.run.summary['Time'] = total_time


with torch.no_grad():
    u_pred = npde_net(torch_tensor_grad(X_star)).cpu().detach().numpy()
    
   
u_pred = np.reshape(u_pred, np.shape(Exact))
u_actual = Exact

u_initial = u_pred


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
    
    ax.set_xlim(( -8, 8))
    ax.set_ylim((-1.5, 1.5))
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


def animate_retrain(u_pred, u_initial, name, time_window):
    
    def update_plot(ii, x, t, u_actual, u_pred, line1, line2, ax):
        line1.set_data(x, u_actual[ii])
        line2.set_data(x, u_initial[ii])
        line3.set_data(x, u_pred[ii])
        mse = np.round(np.mean((u_actual[ii]-u_pred[ii])**2), 3)
        time_text.set_text("Time : " + str(t[ii]))
        loss_text.set_text("Loss : " + str(mse))
        time_retrain.set_text("Time Window retrained for :" + str(time_window))
    
        return [line1,line2, line3, time_text, time_text, loss_text]
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    
    line1, = ax.plot([], [], lw=1.5, label="Actual")
    line2, = ax.plot([], [], lw=1.5, label="Initial Prediction")
    line3, = ax.plot([], [], lw=1.5, label="Retrain Prediction")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    loss_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    time_retrain = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    
    ax.set_xlim((-8, 8))
    ax.set_ylim((-1.5, 1.5))
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


# %%

animate(u_pred, "Initial")

mse_time = []
for it, val in enumerate(t): 
    mse_time.append(np.mean((u_actual[it]-u_pred[it])**2))
    
mse_time = np.asarray(mse_time)
plt.figure()
plt.plot(t, mse_time)

wandb.log({'MSE across the time domain : Initial': wandb.Image(plt)})

# %%

'''
max_error_time_idx = np.argmax(mse_time)
max_error_time = t[max_error_time_idx][0]

X_f_t_torch = torch_tensor_grad(uniform_sampler([max_error_time-2, -8], [max_error_time, 8], 2, 1000))
X_f_torch = torch.cat([X_f_torch, X_f_t_torch])

wandb.run.summary['Time Domain - Overfitted '] = np.asarray([max_error_time-2, max_error_time])


optimizer = torch.optim.Adam(npde_net.parameters(), lr=learning_rate)

start_time = time.time()

epochs = configuration['Retrain Epochs']

for it in tqdm(range(epochs)):
    optimizer.zero_grad()
    
    domain_loss = npde_loss(X_f_torch)
    initial_loss = recon_loss(X_i_torch, u_i_torch)    
    boundary_loss = BC(X_lb_torch, X_ub_torch)

    loss = domain_loss + initial_loss + boundary_loss
    
    loss.backward()
    optimizer.step()
    
    print('Total.  It: %d, Loss: %.3e' % (it, loss.item()))
    
def closure():    
    optimizer.zero_grad()
         
    domain_loss = npde_loss(X_f_torch)
    loss = domain_loss 

    print('QN.  It: %d, Loss: %.3e' % (it, loss.item()))
    loss.backward()
    return loss
    

for ii in tqdm(range(20)):
    optimizer.step(closure)
    
with torch.no_grad():
    u_pred = npde_net(torch_tensor_grad(X_star)).detach().numpy()
    
   
u_pred = np.reshape(u_pred, np.shape(Exact))
u_actual = Exact

animate_retrain(u_pred, u_initial, "Retrain_Initial_Data_and_temporal_domain", (max_error_time-2,max_error_time))


    
# %%
mse_time = []
for it, val in enumerate(t): 
    mse_time.append(np.mean((u_actual[it]-u_pred[it])**2))
    
mse_time = np.asarray(mse_time)
plt.figure()
plt.plot(t, mse_time)

wandb.log({'MSE across the time domain : Retrained only over the weak time domain': wandb.Image(plt)})

'''