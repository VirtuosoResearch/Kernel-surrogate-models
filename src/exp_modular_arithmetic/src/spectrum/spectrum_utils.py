import matplotlib.pyplot as plt
import math
from datetime import datetime
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Pytorch's efficient attention doesn't allow Hessian computation by default
from torch.nn.attention import SDPBackend, sdpa_kernel

from scipy.stats import norm
import yaml

def plot_spectral_density(flat_eigen, flat_weight, sigma=0.01, grid_size=100, plot_individual=False, file_label='', label='Spectral Density'):
    # Determine the range for the lambda grid
    lambda_min = min(flat_eigen) - 1.0
    lambda_max = max(flat_eigen) + 1.0
    #lambda_min = -4
    #lambda_max = 4
    
    # Create a lambda grid
    lambdas = np.linspace(lambda_min, lambda_max, grid_size)
    delta_lambda = lambdas[1] - lambdas[0]
    
    # Initialize the total density
    total_density = np.zeros_like(lambdas)
    
    #plt.figure(figsize=(10, 6))
    
    if plot_individual:
        # Plot individual Gaussian contributions
        for eig, w in zip(flat_eigen, flat_weight):
            gaussian = w * norm.pdf(lambdas, loc=eig, scale=sigma)
            plt.plot(lambdas, gaussian, color='gray', ls="dotted", alpha=0.3)
            total_density += gaussian
    else:
        # Sum all contributions without plotting individual ones
        for eig, w in zip(flat_eigen, flat_weight):
            total_density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    
    # Normalize the total density
    total_density /= np.sum(total_density) * delta_lambda
    
    # Plot the total spectral density
    plt.plot(lambdas, total_density, color='red', linewidth=1, ls="dotted", label=label)
    plt.hist(flat_eigen, bins=50, weights=flat_weight, alpha=0.5, density=True, label='SLQ', color='red')
    
    plt.xlabel('Eigenvalue (λ)')
    plt.ylabel('Density')
    plt.yscale("log", base=10)
    plt.ylim(bottom=1e-10)
    plt.title(label)
    plt.legend()
    plt.grid(True)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    time_str = f"time: {formatted_time}"
    plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
    plt.savefig(f"spectrum/figs/{file_label}/{label}_SLQ.png", dpi=150)
    plt.draw()
    plt.close()


def get_true_curve(grid, eigenvalues):
    left_boundary = np.mean(np.min(eigenvalues, axis = 1))-1
    right_boundary= np.mean(np.max(eigenvalues, axis = 1)) +1
    n_grid = 100
    grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
    curve = []
    for t in grid:
        density = gaussian_density(t, eigenvalues)
        value = np.mean(density)
        curve.append(value)
    return curve
    

def gaussian_density(t, values):
    sigma=0.01
    coeff = 1.0 / np.sqrt(2 * math.pi * sigma**2)
    val = -(values - t) ** 2
    val = val / (2.0 * sigma**2)
    val = np.exp(val)
    density = coeff * val
    return density

def loss_fn(logits, sequence, reduction='mean'):
    return F.cross_entropy(logits[:, -1], sequence[:, -1], reduction=reduction)

def get_hessian(loss, model):
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])
    H = torch.zeros(n_params, n_params, dtype=loss.dtype, device=loss.device)
    for i in range(n_params):
        second_grads = torch.autograd.grad(grad_flat[i], params, retain_graph=True)
        second_grads_flat = torch.cat([sg.contiguous().view(-1) for sg in second_grads])
        H[:, i] = second_grads_flat
    return H

def exact_hessian_eigvals(model, loss):
    H = get_hessian(loss, model)
    eigvals, eigvecs = torch.linalg.eigh(H)
    return eigvals, eigvecs, H

class real_spectrum_calculator():
    def __init__(self, model, loss, layer_names, weights, model_name='', device='cpu'):
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.layer_names = layer_names
        self.weights = weights
        for w in self.weights:
            w = w.to(device)
        self.model_name = model_name
        self.device = device
        
        folder_path = f'./spectrum/logs/{self.model_name}'
        self.folder_path = folder_path

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"create: {folder_path}")


    def get_hessian_by_layer(self):
        layer_hessians = {}
        loss = self.loss

        # Iterate over immediate submodules (layers) of the model
        for l in range(len(self.weights)):
            
            weight = self.weights[l]
            print(weight.shape)
            #print(name)
            #params = list(layer.parameters())
            #if not params:
            #    continue  # Skip layers without parameters

            # Sum of parameters in the current layer
            #n_params_layer = sum(p.numel() for p in params)
            n_params_layer = weight.numel()
            print(f"Computing Hessian of layer: {self.layer_names[l]} with parameters: {n_params_layer}")

            # Compute first-order gradients for the layer's parameters
            #grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grads = torch.autograd.grad(loss, weight, create_graph=True)
            grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])

            # Initialize a Hessian matrix for the current layer
            H_layer = torch.zeros(n_params_layer, n_params_layer,
                                dtype=torch.float16, device=loss.device)

            # Compute second-order derivatives for each parameter element of the layer
            for i in tqdm(range(n_params_layer)):

                # Compute second derivatives of grad_flat[i] w.r.t. the layer parameters
                #second_grads = torch.autograd.grad(grad_flat[i], params, retain_graph=True, create_graph=True)
                second_grads = torch.autograd.grad(grad_flat[i], weight, create_graph=True)
                second_grads_flat = torch.cat([sg.contiguous().view(-1) for sg in second_grads]).to(torch.float16)
                H_layer[:, i] = second_grads_flat

            H_layer = H_layer.to('cpu')
            # Store the Hessian block for the current layer
            layer_hessians[self.layer_names[l]] = H_layer
            #break

        return layer_hessians

    def exact_hessian_eigvals_by_layer_(self):
        # Compute Hessians for each layer
        layer_hessians = self.get_hessian_by_layer()

        eigvals_by_layer = {}
        eigvecs_by_layer = {}

        # Compute eigenvalues and eigenvectors for each layer's Hessian
        for name, H in layer_hessians.items():
            eigvals, eigvecs = torch.linalg.eigh(H)
            eigvals_by_layer[name] = eigvals
            eigvecs_by_layer[name] = eigvecs

        self.save_all(eigvals_by_layer, eigvecs_by_layer, layer_hessians)

        return eigvals_by_layer, eigvecs_by_layer, layer_hessians
    
    def exact_hessian_eigvals(self, file_list):
        self.exact_hessian()
        return self.exact_eigvals()

    def exact_hessian(self):
        # Compute Hessians for each layer
        layer_hessians = self.get_hessian_by_layer()
        self.save_hessian(layer_hessians)

        return layer_hessians

    def save_hessian(self, layer_hessian):
        for name, H in layer_hessian.items():
            with open(f"{self.folder_path}/{name}_exact_H.txt", 'w') as f:
                yaml.dump(H.detach().cpu().numpy().tolist(), f)
    
    def load_hessian(self, file_list):
        # load Hessian
        layer_hessians = {}
        for name in file_list:
            with open(f"{self.folder_path}/{name}_exact_H.txt", 'r') as f:
                H = yaml.safe_load(f)
            H = np.array(H)
            layer_hessians[name] = H

        return layer_hessians
    
    # compute eigenvalues by power method
    def power_eigvals(self, use_local=False):
        # get H matrix
        if use_local:
            layer_hessians = self.load_hessian(file_list=self.layer_names)
        else:
            layer_hessians = self.get_hessian_by_layer()
            self.save_hessian(layer_hessians)

    def exact_eigvals(self, use_local=True):
        # load Hessian
        # get H matrix
        if use_local:
            layer_hessians = self.load_hessian(file_list=self.layer_names)
        else:
            layer_hessians = self.get_hessian_by_layer()
            self.save_hessian(layer_hessians)

        eigvals_by_layer = {}
        eigvecs_by_layer = {}
        for name, H in layer_hessians.items():
            print(f"Computing eigenvalues of layer: {name}")
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals_by_layer[name] = eigvals
            eigvecs_by_layer[name] = eigvecs
        
        self.save_eigvals(eigvals_by_layer)
        
        return eigvals_by_layer, eigvecs_by_layer, layer_hessians

    def save_eigvals(self, eigvals_by_layer):
        for name, eigvals in eigvals_by_layer.items():
            # Save the eigenvalues and eigenvectors
            with open(f"{self.folder_path}/{name}_exact_eigvals.txt", 'w') as f:
                yaml.dump(eigvals.tolist(), f)

    def load_eigenvals(self, file_list):
        eigvals_by_layer = {}
        for name in file_list:
            with open(f"{self.folder_path}/{name}_exact_eigvals.txt", 'r') as f:
                eigvals = yaml.safe_load(f)
            eigvals = np.array(eigvals)
            eigvals_by_layer[name] = eigvals

        return eigvals_by_layer


def load_eigenvals(file):
    with open(file, 'r') as f:
        eigvals = yaml.safe_load(f)
    eigvals = np.array(eigvals)

    return eigvals
