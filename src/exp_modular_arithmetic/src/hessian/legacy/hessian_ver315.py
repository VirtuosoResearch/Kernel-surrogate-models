#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICUnp.linalgR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import copy

#from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from scipy.stats import norm
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.stats import pearsonr

from src.bound.compute_bounds import *

def compute_model_norm(model, p=2):
    norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in model.parameters() if p.requires_grad]), 2)
    return norm

def load_batch_func(batch, device='cpu'):
    batch = batch[0].to(device)
    inputs = batch[:, :-1]
    targets = batch
    batch_size = batch.shape[0]
    return inputs, targets, batch_size

def filter_eigenvalues(eigen_list, weight_list, threshold=None):
    filtered_eigen = []
    filtered_weight = []
    #print(np.max(weight_list))
    for eig, w in zip(eigen_list, weight_list):
        if threshold is not None:
            if eig >= threshold and w >= 1e-7:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
        else:
            if w >= 1e-10:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
    #print(filtered_eigen)
    return filtered_eigen, filtered_weight

def renormalize_weights(filtered_weight, epsilon=1e-12):
    total = sum(filtered_weight)
    if total > 0:
        renormalized_weight = [w / (total + epsilon) for w in filtered_weight]
    else:
        # Handle case where all weights are zero
        renormalized_weight = [0.0 for _ in filtered_weight]
    return renormalized_weight

def construct_spectral_density(flat_eigen, flat_weight, lambdas, sigma=0.1):
    density = np.zeros_like(lambdas)
    for eig, w in zip(flat_eigen, flat_weight):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    
    # Normalize the density
    density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
    density /= density_sum + 1e-12  # Avoid division by zero
    return density

def flat_list(original_list):
    return [float(e) for run in original_list for e in run]

def list_aggregate(list_a, list_b, batch_size):
    if len(list_a) == 0:
        list_a = [float(b) * batch_size for b in list_b]
    else:
        list_a = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
    return list_a

def compute_spectral_divergences(
    eigen_list_1, weight_list_1, 
    eigen_list_2, weight_list_2,
    measure = 'kl',
    sigma=0.1, grid_size=100
):
    # Step 1: Validate Inputs
    if not (len(eigen_list_1) == len(weight_list_1)):
        raise ValueError("eigen_list_1 and weight_list_1 must have the same number of SLQ runs.")
    if not (len(eigen_list_2) == len(weight_list_2)):
        raise ValueError("eigen_list_2 and weight_list_2 must have the same number of SLQ runs.")
    
    # Step 2: Flatten the eigenvalues and weights
    
    # Step 3: Determine the global min and max eigenvalues for the common grid
    all_eigen = eigen_list_1 + eigen_list_2
    lambda_min = min(all_eigen) - 1.0  # Padding to ensure coverage
    lambda_max = max(all_eigen) + 1.0
    
    # Step 4: Create a common lambda grid
    common_lambdas = np.linspace(lambda_min, lambda_max, grid_size)
    delta_lambda = common_lambdas[1] - common_lambdas[0]
    
    # Step 5: Construct spectral densities using Gaussian kernels
    def construct_density(eigen_list, weight_list, lambdas, sigma):
        density = np.zeros_like(lambdas)
        for eig, w in zip(eigen_list, weight_list):
            density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
        # Normalize the density
        density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
        density /= density_sum + 1e-12  # Avoid division by zero
        return density
    
    density_1 = construct_density(eigen_list_1, weight_list_1, common_lambdas, sigma)
    density_2 = construct_density(eigen_list_2, weight_list_2, common_lambdas, sigma)
    
    # Step 6: Compute KL Divergence (D_KL(P || Q))
    # To ensure numerical stability, add a small epsilon where necessary
    #print(density_1)
    #print(density_2)
    epsilon = 1e-12
    p = density_1 + epsilon
    q = density_2 + epsilon
    if measure == 'kl':
        divergence = np.sum(p * np.log(p / q)) * delta_lambda
    elif measure == 'js':
        # Step 7: Compute Jensen-Shannon Divergence (D_JS(P || Q))
        m = 0.5 * (p + q)
        d_kl_p_m = np.sum(p * np.log(p / m)) * delta_lambda
        d_kl_q_m = np.sum(q * np.log(q / m)) * delta_lambda
        divergence = 0.5 * (d_kl_p_m + d_kl_q_m)
        
    return divergence, common_lambdas


def create_spectral_density(eigen_list_full, weight_list_full, sigma=0.1, grid_size=1000):
    eigen_values = [eig for run in eigen_list_full for eig in run]
    weights = [w for run in weight_list_full for w in run]
    lambda_min = min(eigen_values) - 1
    lambda_max = max(eigen_values) + 1
    print("min, max: ", lambda_min, lambda_max)
    lambdas = np.linspace(lambda_min, lambda_max, grid_size)
    density = np.zeros_like(lambdas)
    for eig, w in zip(eigen_values, weights):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    density /= np.sum(density) * (lambdas[1] - lambdas[0])
    return lambdas, density

def kl_divergence(p, q, lambdas):
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q)) * (lambdas[1] - lambdas[0])

def js_divergence(p, q, lambdas):
    m = 0.5 * (p + q)
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    m = m + epsilon
    D_KL_P_M = np.sum(p * np.log(p / m)) * (lambdas[1] - lambdas[0])
    D_KL_Q_M = np.sum(q * np.log(q / m)) * (lambdas[1] - lambdas[0])
    D_JS = 0.5 * (D_KL_P_M + D_KL_Q_M)
    return D_JS

def total_variation(p, q, lambdas):
    return 0.5 * np.sum(np.abs(p - q)) * (lambdas[1] - lambdas[0])


def normalization_(vs, epsilon=1e-6):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    """
    norms = [torch.sum(v*v) for v in vs]
    norms = [(norm**0.5).cpu().item() for norm in norms]
    vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
    return vs
    """
    return [v / (torch.norm(v) + epsilon) for v in vs]

def orthnormal_(ws, vs_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for vs in vs_list:
        for w, v in zip(ws, vs):
            w.data.add_(-v*(torch.sum(w*v)))
    return normalization(ws)


def sqrt_with_neg_handling(arr):
    result = np.where(arr < 0, 0, np.sqrt(arr))
    return result

class Hessian_Calculator():
    def __init__(self, model, loss_fn, p, dataloader=None, valid_dataloader=None, external_load_batch_func=None, device='cpu'):
        self.p = p
        self.num_classes = p+2
        self.model = model.eval()  # make model is in evaluation model
        self.loss_fn = loss_fn
        self.aggregate_method = 'mean'

        if external_load_batch_func is not None:
            self.load_batch_func = external_load_batch_func
        else:
            self.load_batch_func = load_batch_func
        
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device

        # get splited weights
        self.layers = get_layers(self.model)
        self.weights, self.layer_names, self.grouped_layer_weights, self.grouped_layer_names = get_grouped_layer_weights(self.model)
        print(self.layer_names)
        #print(self.weights[0].grad)

        self.hessian_norms = []
        self.layer_trace = []
        self.lambda_max_list = []

        
        self.spectrum_divergence_list = []
        self.spectrum_entropy_list = []
        self.weighted_entropy_list = []
        self.centroid_list = []
        self.spread_list = []
        self.effective_rank_list = []
        self.stable_rank_list = []
        self.lambda_max_list = []
        self.condition_list = []

        self.valid_spectrum_entropy_list = []
        self.valid_weighted_entropy_list = []
        self.valid_centroid_list = []
        self.valid_spread_list = []
        self.valid_effective_rank_list = []
        self.valid_stable_rank_list = []
        self.valid_lambda_max_list = []
        self.valid_condition_list = []

        self.max_eigenvector_1 = None
        self.lambda_1 = 0

        self.noise_sensitivity = 0

        self.sample_layer = ['head.weight']
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def list_aggregate(self, list_a, list_b, batch_size=0, aggregate_method='mean'):
        if aggregate_method == 'mean':
            if len(list_a) == 0:
                list_result = [float(b) * batch_size for b in list_b]
            else:
                list_result = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
            return list_result

        elif aggregate_method == 'max':
            if len(list_a) == 0:
                list_result = list_b
            else:
                list_result = [max(a, b) for a, b in zip(list_a, list_b)]
            # Return element-wise maximum from list_a and list_b
            return list_result

        elif aggregate_method == 'min':
            if len(list_a) == 0:
                list_result = list_b
            else:
                list_result = [min(a, b) for a, b in zip(list_a, list_b)]
            # Return element-wise minimum from list_a and list_b
            return list_result

        else:
            raise ValueError(f"Unknown method: {aggregate_method}")
    
    def group_div_const(self, X, c):
        return [x/c for x in X]
    
    def hessian_quadratic_form(self, model, loss, noise_vector):
        
        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])
        
        # Compute the dot product between gradients and noise vector.
        grad_dot_noise = torch.dot(grad_vector, noise_vector)
        
        # Compute Hessian-vector product using the Pearlmutter trick.
        Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
        Hv_vector = torch.cat([h.reshape(-1) for h in Hv])
        
        # The quadratic form δ^T H δ.
        quad_form = torch.dot(noise_vector, Hv_vector)
        return quad_form
    
    def hessian_quadratic_2_form(self, model, loss, noise_vector):
        
        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])

        noise_tuple = []
        idx = 0
        for g in grads:
            sz = g.numel()
            noise_tuple.append(noise_vector[idx: idx + sz].view_as(g))
            idx += sz
        
        # Compute the dot product between gradients and noise vector.
        #grad_dot_noise = torch.dot(grad_vector, noise_vector)
        
        # Compute Hessian-vector product using the Pearlmutter trick.
        #Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
        #HHv = torch.autograd.grad(Hv, model.parameters(), retain_graph=True)
        Hv = torch.autograd.grad(grads, model.parameters(), grad_outputs=noise_tuple, retain_graph=True)    
        HHv = torch.autograd.grad(grads, model.parameters(), grad_outputs=Hv, retain_graph=True)
        HHv_vector = torch.cat([h.reshape(-1) for h in HHv])
        
        # The quadratic form δ^T H δ.
        quad_form = torch.dot(noise_vector, HHv_vector)
        return quad_form

    def compare_hessian(self, logger, log_i, train_num, valid_num):
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            sample_num = 50
            train_hessian_list = []
            train_hessian_2_list = []
            valid_hessian_list = []
            valid_hessian_2_list = []
            for i in range(sample_num):
                noise_vector = None
                train_hessian = 0
                train_hessian_2 = 0
                for train_batch in self.dataloader:
                    data, target, batch_size = self.load_batch_func(train_batch, device)
                    output = model(data)
                    loss = loss_fn(output, target)

                    # Compute gradients to get the shape.
                    if noise_vector is None:
                        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                        grad_vector = torch.cat([g.reshape(-1) for g in grads])

                        # Sample the noise vector once.
                        #noise_vector = torch.randn_like(grad_vector)
                        noise_vector = torch.randint_like(grad_vector, high=2)
                        noise_vector[noise_vector == 0] = -1
                    train_quad = self.hessian_quadratic_form(model, loss, noise_vector)
                    train_H2 = self.hessian_quadratic_2_form(model, loss, noise_vector)
                    train_hessian += train_quad.item()*batch_size
                    train_hessian_2 += train_H2.item()*batch_size
                train_hessian /= train_num
                train_hessian_2 /= train_num

                valid_hessian = 0
                valid_hessian_2 = 0
                for valid_batch in self.valid_dataloader:
                    data, target, batch_size = self.load_batch_func(valid_batch, device)
                    output = model(data)
                    loss = loss_fn(output, target)

                    valid_quad = self.hessian_quadratic_form(model, loss, noise_vector)
                    valid_hessian += valid_quad.item()*batch_size
                    valid_H2 = self.hessian_quadratic_2_form(model, loss, noise_vector)
                    valid_hessian_2 += valid_H2.item()*batch_size
                valid_hessian /= valid_num
                valid_hessian_2 /= valid_num

                noise_vector = None
                train_hessian_list.append(train_hessian)
                train_hessian_2_list.append(train_hessian_2)
                valid_hessian_list.append(valid_hessian)
                valid_hessian_2_list.append(valid_hessian_2)
            train_hessian = np.mean(train_hessian_list)
            train_hessian_2 = np.mean(train_hessian_2_list)
            valid_hessian = np.mean(valid_hessian_list)
            valid_hessian_2 = np.mean(valid_hessian_2_list)

        self.train_hessian = train_hessian
        self.train_hessian_2 = train_hessian_2
        self.valid_hessian = valid_hessian
        self.valid_hessian_2 = valid_hessian_2

        logger.log("train_hessian", train_hessian, log_i)
        logger.log("train_hessian_2", train_hessian_2, log_i)
        logger.log("valid_hessian", valid_hessian, log_i)
        logger.log("valid_hessian_2", valid_hessian_2, log_i)

        plot_curves(logger, ['train_hessian', 'train_hessian_2', 'valid_hessian', 'valid_hessian_2'], path_name='hessian')
        #plot_curves(logger, ['train_hessian_2', 'valid_hessian_2'], path_name='hessian_2')

        return train_hessian, train_hessian_2, valid_hessian, valid_hessian_2
            
    def check_slq(self, logger, i, train_num, valid_num, n_iter=100, n_v=1):
        with sdpa_kernel(SDPBackend.MATH):
            print("=======> SLQ for full model")
            values_full, weights_full = self.get_full_spectrum(n_iter=n_iter, n_v=n_v, dataloader=self.dataloader)
            self.values_full = values_full.tolist()
            self.weights_full = weights_full.tolist()
            d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.slq_H_trace = np.sum(values_full * weights_full) * d
            self.slq_H2_trace = np.sum(values_full**2 * weights_full)* d
            self.hvp_H_trace, self.hvp_H2_trace, _, _ = self.compare_hessian(logger, i, train_num, valid_num)
            print(self.slq_H_trace, self.slq_H2_trace, self.hvp_H_trace)
            slq_lambda_max = max(values_full)

            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            
            train_lambda_max = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(loss.mean(), model, power_iter=100)
                train_lambda_max += batch_lambda_max * batch_size
            train_lambda_max /= len(self.dataloader.dataset)

        logger.log("slq_H_trace", self.slq_H_trace, i)
        logger.log("slq_H2_trace", self.slq_H2_trace, i)
        logger.log("hvp_H_trace", self.hvp_H_trace, i)
        logger.log("hvp_H2_trace", self.hvp_H2_trace, i)
        data_names = ['slq_H_trace', 'hvp_H_trace']
        plot_curves(logger, data_names, path_name='check', file_name='hessian')
        data_names = ['slq_H2_trace', 'hvp_H2_trace']
        plot_curves(logger, data_names, path_name='check', file_name='hessian_2')

        logger.log("hvp_lambda_max", train_lambda_max, i)
        logger.log("slq_lambda_max", slq_lambda_max, i)
        plot_curves(logger, ['hvp_lambda_max', 'slq_lambda_max'], path_name='check', file_name='lambda_max')


    def collect(self, train_num, valid_num):
        self.trace_based_measure()
        #self.batch_collect()
        self.batch_aggregate(train_num, valid_num)

    def trace_based_measure(self, device = "cpu", maxIter=100, tol=1e-3):
        self.layer_trace = []
        self.hessian_norms = []
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            
            for batch in self.dataloader:
                # Specific data process, in order to fit the loss input
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                model.zero_grad()

                # Trace of Hessian. Save in ./results/hessian
                grouped_layer_trace = self.compute_hessians_trace(model, loss.mean(), batch_size, device)
                self.layer_trace = self.list_aggregate(self.layer_trace, grouped_layer_trace, batch_size, aggregate_method='mean')
                #print("method 0: ", self.layer_trace)

                # stable rank
                stable_rank_list, lambda_max_list = self.compute_stable_rank(loss.mean(), batch_size)
                #print("method 1: ", lambda_max_list)
                self.stable_rank_list = self.list_aggregate(self.stable_rank_list, stable_rank_list, batch_size, aggregate_method='mean')
                self.lambda_max_list = self.list_aggregate(self.lambda_max_list, lambda_max_list, batch_size, aggregate_method='mean')

                # max eigenvalue
                #self.compute_eigenvalues(loss.mean(), batch_size)

                # Hessian bound
                layer_hessian_quantities = self.compute_generalization_bound(model, loss.mean(), self.device)
                self.hessian_norms = self.list_aggregate(self.hessian_norms, layer_hessian_quantities, batch_size, aggregate_method='max')

                #self.batch_spectral_density(loss.mean(), batch_size)
                #self.compute_sensitivity(loss.mean(), data, target, batch_size)
                
            # Compute the spectral density
            #self.spectral_density(n_iter=100, n_v=5)

            #self.compute_generalization_bound_2(model)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None   

    def batch_collect(self):
        self.layer_trace = []
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            for batch in self.dataloader:
                # Specific data process, in order to fit the loss input
                #batch = batch[0].to(device)
                #data = batch[:, :-1]
                #target = batch
                #batch_size = batch.shape[0]
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                model.zero_grad()
                # item_1: f(x)f(x)^T, item_2: diag(p) - pp^T. Save in ./results/prob
                #self.compute_item(model, data, target, batch_size)

                # sensitivity: inject noise to input, and estimate the difference of loss. Save in ./results/input
                #self.compute_sensitivity(model, loss_fn, data, target, batch_size)

                # The ratio of the first singular value and the second singular value of loss. Save in ./results/distance
                #self.compute_singular_ratio(model, loss.mean(), batch_size)

                # Compute the trace of Hessian of weight decay
                #self.compute_wd_hessians_trace()

                # Compute the eigenvalues
                #self.compute_eigenvalues(model, loss.mean(), batch_size)

                # Trace of Hessian. Save in ./results/hessian
                self.compute_hessians_trace(model, loss.mean(), batch_size, device)

                #self.compute_stable_rank_2(model, loss.mean(), batch_size, device)
                self.compute_stable_rank_per_group(loss.mean(), batch_size)

                # Hessian bound
                #self.compute_generalization_bound(model, loss.mean(), self.device)

                #self.batch_spectral_density(loss.mean(), batch_size)
                
            # Compute the spectral density
            #self.spectral_density(n_iter=100, n_v=5)

            self.compute_generalization_bound_2(model)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None   
    
    def batch_aggregate(self, train_num, valid_num):
        # trace based measure
        self.layer_trace = self.group_div_const(self.layer_trace, train_num)
        #self.stable_rank_list = self.group_div_const(self.stable_rank_list, train_num)
        self.lambda_max_list = self.group_div_const(self.lambda_max_list, train_num)


        #self.stable_rank_per_group = self.group_div_const(self.stable_rank_per_group, train_num)
        #print(self.layer_trace)
        #print(self.effective_rank_list)
        #self.shapescale = (self.layer_trace*np.array(self.effective_rank_list)).tolist()
        #print(self.shapescale)
        #print(self.layer_trace)
        #for i in range(len(self.layer_trace)):
        #    self.layer_trace[i]  = float(self.layer_trace[i])
        #self.spectral_density = self.group_div_const(self.spectral_density, train_num)
        #print(self.spectrum_divergence_list)
        #print(self.spectrum_entropy_list)
        #self.spectrum_divergence_list = self.group_div_const(self.spectrum_divergence_list, train_num)
        #self.centroid_list = self.group_div_const(self.centroid_list, train_num)
        #self.spread_list = self.group_div_const(self.spread_list, train_num)
        #self.weighted_entropy_list = self.group_div_const(self.weighted_entropy_list, train_num)
        #self.spectrum_entropy_list = self.group_div_const(self.spectrum_entropy_list, train_num)

        #self.item_1 /= train_num
        #self.item_2 /= train_num
        #self.cosine_similarities = np.mean(self.cosine_similarities / train_num)
        #self.grad_norms = np.mean(self.grad_norms / train_num)
        #self.wd_grad_norms = np.mean(self.wd_grad_norms / train_num)
        #self.l2_distance = np.mean(self.l2_distance / train_num)
        #print(self.cosine_similarities)
        #self.trace = self.layer_trace / train_num
        #print(self.trace)
        #self.lambda_1 = self.lambda_1 / train_num
        #self.condition = self.lambda_1 / self.trace
        #print("condition: ", self.condition)
        """
        self.lambda_1 = self.lambda_1 / train_num
        self.lambda_2 = self.lambda_2 / train_num
        self.lambda_1 = np.mean(self.lambda_1)
        self.lambda_2 = np.mean(self.lambda_2)
        #self.lambda_n = np.mean(self.lambda_n/train_num)
        self.condition = self.lambda_1 / self.lambda_2
        eigen_distance = []
        for i in range(len(self.max_eigenvector_1)):
            eigen_distance.append(torch.norm(self.max_eigenvector_1[i] - self.max_eigenvector_2[i]).item())
        self.max_eigenvector_1 = [eigenvector / train_num for eigenvector in self.max_eigenvector_1]
        self.max_eigenvector_2 = [eigenvector / train_num for eigenvector in self.max_eigenvector_2]
        for i in range(len(self.max_eigenvector_1)):
            eigen_distance.append(torch.norm(self.max_eigenvector_1[i] - self.max_eigenvector_2[i]).item())
        #self.condition = np.mean(self.lambda_1 / ((self.trace - self.lambda_1)/(self.trace_num-1)))
        self.eigenvector_distance = np.mean(eigen_distance)
        """
        #self.wd_trace = np.mean(self.layer_wd_trace)
        #self.trace = np.mean(self.trace)

        #self.noise_sensitivity /= train_num
        #print(self.noise_sensitivity)
        
        #self.loss_singularvector_distance = np.mean(self.loss_singularvector_distance / train_num)
        #self.loss_singularvalue_distance = np.mean(self.loss_singularvalue_distance / train_num)

        hessian_quantities = np.sum(sqrt_with_neg_handling(np.array(self.hessian_norms))) / np.sqrt(train_num)
        self.train_hessianmeasurement = (hessian_quantities).item()


    def compute_hessians_trace(self, model, loss, batch_size, aggregate_method='mean', device = "cpu", maxIter=100, tol=1e-8):
        # Get parameters and gradients of corresponding layer
        grouped_layer_trace = []
        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

            layer_traces = []
            trace_vhv = []
            trace = 0.
            # Start Iterations
            for _ in range(maxIter):
                vs = [torch.randint_like(weight, high=2) for weight in weights]
                    
                # generate Rademacher random variables
                for v in vs:
                    v[v == 0] = -1

                model.zero_grad()  
                Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])
                #tmp_layer_traces = sum([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

                layer_traces.append(tmp_layer_traces)
                trace_vhv.append(sum(tmp_layer_traces))

                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                    print("break hessian trace")
                    break
                else:
                    trace = np.mean(trace_vhv)
            layer_trace = np.mean(np.array(layer_traces), axis=0)
            #grouped_layer_trace.append(np.sum(layer_trace, axis=0))
            grouped_layer_trace.append(trace)
        #print(grouped_layer_trace)
        return grouped_layer_trace
        #self.layer_trace = self.list_aggregate(self.layer_trace, grouped_layer_trace, batch_size, aggregate_method=aggregate_method)
        """
        if len(self.layer_trace) == 0:
            #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
            self.layer_trace = grouped_layer_trace
        else:
            self.layer_trace = np.maximum(grouped_layer_trace, self.layer_trace).tolist()
        """
        #print(self.layer_trace)

    
    # only support top_n=1
    def compute_eigenvalues(self, loss, batch_size, top_n=1, maxIter=100, tol=1e-8):
        model = self.model
        #weights = self.weights
        
        topn_eigenvalues_list = []
        eigenvectors_list = []
        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

            topn_eigenvalues = []
            eigenvectors = []
            computed_dim = 0
            while computed_dim < top_n:
                
                eigenvalues = None
                vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
                vs = normalization(vs)  # normalize the vector

                for _ in range(maxIter):
                    #vs = orthnormal(vs, eigenvectors)
                    #model.zero_grad()

                    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                    tmp_eigenvalues = sum([ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

                    vs = normalization(Hvs)

                    if eigenvalues == None:
                        eigenvalues = tmp_eigenvalues
                    else:
                        if abs(eigenvalues - tmp_eigenvalues) / (abs(eigenvalues) + 1e-8) < tol:
                            break
                        else:
                            eigenvalues = tmp_eigenvalues
                topn_eigenvalues.append(eigenvalues)
                eigenvectors.append(vs)
                computed_dim += 1
            topn_eigenvalues_list.append(topn_eigenvalues[0])
            eigenvectors_list.append(eigenvectors)
        """
        topn_eigenvalues_list = []
        eigenvectors_list = []

        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            topn_eigenvalues = []
            eigenvectors = []
            computed_dim = 0

            while computed_dim < top_n:
                eigenvalue_estimate = None
                vs = [torch.randn_like(w) for w in weights]
                vs = normalization_list(vs)

                for _ in range(maxIter):
                    vs = orthnormal_list(vs, eigenvectors)
                    self.model.zero_grad()
                    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                    tmp_eigenvalue = sum(torch.sum(Hv * v).item() for Hv, v in zip(Hvs, vs))
                    vs = normalization_list(Hvs)

                    if eigenvalue_estimate is None:
                        eigenvalue_estimate = tmp_eigenvalue
                    else:
                        rel_change = abs(eigenvalue_estimate - tmp_eigenvalue) / (abs(eigenvalue_estimate) + 1e-8)
                        eigenvalue_estimate = tmp_eigenvalue
                        if rel_change < tol:
                            break

                topn_eigenvalues.append(eigenvalue_estimate)
                eigenvectors.append(vs)
                computed_dim += 1

            topn_eigenvalues_list.append(topn_eigenvalues[0])
            eigenvectors_list.append(eigenvectors)
        """
        #print(topn_eigenvalues_list)
        # Max eigenvalue. In process
        max_eigenvalue, max_eigenvector = compute_eigenvalue(self.model, loss, self.device, top_n=1)
        max_eigenvector = max_eigenvector[0]
        #print(max_eigenvector[0].shape)
        #block_sim_1 = F.cosine_similarity(max_eigenvector[0], torch.eye(max_eigenvector[0].shape[1]).to(device))
        #block_sim_2 = F.cosine_similarity(max_eigenvector[1], torch.eye(max_eigenvector[1].shape[1]).to(device))
        #block_sim_3 = F.cosine_similarity(max_eigenvector[2], torch.eye(max_eigenvector[2].shape[1]).to(device))
        #block_sim_1 = max_eigenvector[0][:, 0]
        #block_sim_2 = max_eigenvector[1][:, 0]
        #block_sim_3 = max_eigenvector[2][:, 0]
        #self.block_sim_1 += block_sim_1.mean()*batch_size
        #self.block_sim_2 += block_sim_2.mean()*batch_size
        #self.block_sim_3 += block_sim_3.mean()*batch_size
        #print(block_sim_1.shape)
        #print(block_sim_1, block_sim_2, block_sim_3)
        #min_eigenvalue, min_eigenvector = compute_eigenvalue(model, -loss, device, top_n=1)
        #print(max_eigenvalue, min_eigenvalue)
        
        #max_eigenvalue, max_eigenvector = compute_eigenvalue(model, loss, device, top_n=1)
        max_eigenvector_1 = max_eigenvector[0]
        #max_eigenvector_2 = max_eigenvector[1]
        if self.max_eigenvector_1 is None:
            self.max_eigenvector_1 = max_eigenvector_1 * batch_size
            #self.max_eigenvector_2 = max_eigenvector_2 * batch_size
        else:
            self.max_eigenvector_1 += max_eigenvector_1 * batch_size
            #self.max_eigenvector_2 += max_eigenvector_2 * batch_size
        lambda_1 = np.array(max_eigenvalue[0])
        #lambda_2 = np.array(max_eigenvalue[1])
        #print("method 2: ", topn_eigenvalues_list)
        self.lambda_1 += lambda_1 * batch_size
        #self.lambda_2 += lambda_2 * batch_size
        #print(layer_trace, lambda_1)
        #print(estimate_trace, estimate_eigen)
        #self.lambda_n += lambda_n * batch_size

    def compute_generalization_bound(self, model, loss, device="cpu", state_dict = None):
        # Get parameters and gradients of corresponding layer
        weights = self.weights
        gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
        
        vs = []
        for name, module in self.layers.items():
            weight = module.weight
            v = weight.detach().clone() - model.init_state[name+".weight"].to(weight.device)
            vs.append(v)

        model.zero_grad()    
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

        layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
        
        layer_hessian_quantities = np.array(layer_hessian_quantities)
        
        return layer_hessian_quantities
        #layer_hessian_quantities = np.sum(layer_hessian_quantities) / np.sqrt(config.train.hessian_log_size)
        if len(self.hessian_norms) == 0:
            self.hessian_norms = layer_hessian_quantities
        else:
            self.hessian_norms = np.maximum(self.hessian_norms, layer_hessian_quantities)

    def compute_sensitivity(self, loss, data, target, batch_size):
        # noise_sensitivity: Estimate the sensetivity of input. Save in ./results/input
        for i in range(50):
            #noisy_output, noise_norm = model.add_noise_forward(data)
            
            #noise_sensitivity = torch.norm(noisy_output[:, -1] - output[:, -1]) / noise_norm
            #noisy_loss = F.cross_entropy(noisy_output[:, -1], target[:, -1], reduction='none')
            #noise_sensitivity = (noisy_loss - loss) / noise_norm
            noisy_output_1, noisy_output_2, noise_norm = self.model.add_bi_noise_forward(data)
            noisy_loss_1 = F.cross_entropy(noisy_output_1[:, -1], target[:, -1], reduction='none')
            noisy_loss_2 = F.cross_entropy(noisy_output_2[:, -1], target[:, -1], reduction='none')
            noise_sensitivity = (noisy_loss_1 + noisy_loss_2 - 2*loss)
            #noise_sensitivity = (noisy_output_1 + noisy_output_2 - output)[:, -1]

        self.noise_sensitivity += noise_sensitivity.mean().item() * batch_size

    # copy from pyhessian
    def dataloader_hv_product(self, v, weights):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(weight.size()).to(device) for weight in weights]  # accumulate result
        for batch in self.dataloader:
            self.model.zero_grad()
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, device)
            output = self.model(data)
            loss = self.loss_fn(output, target, 'mean')
            loss.backward(create_graph=True)
            #params, gradsH = get_params_grad(self.model) # TODO check
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradients,
                                     weights,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(batch_size) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(batch_size)

        THv = [THv1 / float(num_data) for THv1 in THv]

        return THv
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv
    

    def compute_generalization_bound_2(self, model, device="cpu", state_dict = None):
        # Get parameters and gradients of corresponding layer
        weights = self.weights
        #gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
        
        vs = []
        for name, module in self.layers.items():
            weight = module.weight
            v = weight.detach().clone() - model.init_state[name+".weight"].to(weight.device)
            vs.append(v)

        model.zero_grad()    
        #Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        Hvs = self.dataloader_hv_product(vs, weights)

        layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
        
        layer_hessian_quantities = np.array(layer_hessian_quantities)
        
        #layer_hessian_quantities = np.sum(layer_hessian_quantities) / np.sqrt(config.train.hessian_log_size)
        if len(self.hessian_norms) == 0:
            self.hessian_norms = layer_hessian_quantities
        else:
            self.hessian_norms = np.maximum(self.hessian_norms, layer_hessian_quantities)

    def compute_stable_rank_2(self, model, loss, batch_size, device = "cpu", maxIter=100, tol=1e-3):
        # Get parameters and gradients of corresponding layer
        grouped_layer_trace = []
        for weights in self.grouped_layer_weights:
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

            layer_traces = []
            trace_vhv = []
            trace = 0.
            # Start Iterations
            for _ in range(maxIter):
                vs = [torch.randint_like(weight, high=2) for weight in weights]
                    
                # generate Rademacher random variables
                for v in vs:
                    v[v == 0] = -1

                model.zero_grad()  
                Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
                HHv = torch.autograd.grad(gradients, weights, grad_outputs=Hvs, retain_graph=True)
                tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])
                #tmp_layer_traces = sum([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

                layer_traces.append(tmp_layer_traces)
                trace_vhv.append(sum(tmp_layer_traces))

                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                    break
                else:
                    trace = np.mean(trace_vhv)
            layer_trace = np.mean(np.array(layer_traces), axis=0)
            #grouped_layer_trace.append(np.sum(layer_trace, axis=0))
            grouped_layer_trace.append(trace)
        #print(grouped_layer_trace)
        
        if len(self.layer_trace) == 0:
            #avg_layer_trace = np.mean(np.array(layer_traces), axis=0) / trace_num
            self.layer_trace = grouped_layer_trace
        else:
            self.layer_trace = np.maximum(grouped_layer_trace, self.layer_trace).tolist()

    def approximate_trace_h2(self, loss, weights, maxIter=100, tol=1e-3):
        """
        Approximates trace(H^2) for the given group of weights, 
        where H is the Hessian of 'loss' wrt 'weights'.
        
        Returns: A float approximation of trace(H^2) for that group.
        """
        # 1. First-order gradient wrt 'weights'
        gradients = torch.autograd.grad(loss, weights, 
                                        retain_graph=True, 
                                        create_graph=True)
        
        trace_estimates = []
        trace_running_list = []
        prev_trace_mean = 0.0
        
        for _ in range(maxIter):
            # Generate Rademacher random vectors 'v' for each tensor
            vs = [torch.randint_like(w, high=2) for w in weights]
            for v in vs:
                v[v == 0] = -1  # convert {0,1} to {-1, +1}

            # Step 1: compute H v
            Hv = torch.autograd.grad(gradients, weights, 
                                     grad_outputs=vs,
                                     retain_graph=True)
            
            # Step 2: compute H(Hv) = H^2 v
            HHv = torch.autograd.grad(gradients, weights, 
                                      grad_outputs=Hv,
                                      #grad_outputs=vs,
                                      retain_graph=True)
            
            # Now estimate v^T (H^2 v) = sum( (HHv_i * v_i).sum() ) for each i
            # Each i corresponds to a parameter tensor in this group
            vHv_sum = 0.0
            for (HHv_i, v_i) in zip(HHv, vs):
                vHv_sum += torch.sum(HHv_i * v_i).item()
            
            trace_estimates.append(vHv_sum)
            trace_running_list.append(vHv_sum)
            
            # Early stopping check
            new_mean = np.mean(trace_running_list)
            rel_change = abs(new_mean - prev_trace_mean) / (abs(prev_trace_mean) + 1e-6)
            if rel_change < tol:
                break
            prev_trace_mean = new_mean

        # Final approximate trace(H^2) is the mean of all samples
        return np.mean(trace_estimates)

    def approximate_lambda_max_old(self, loss, weights, power_iter=20):
        """
        Approximates the largest eigenvalue of the Hessian wrt 'weights'
        using power iteration. 
        Only works well if the Hessian is PSD.
        
        Returns: float lambda_max
        """
        # 1. Compute first-order gradient
        gradients = torch.autograd.grad(loss, weights, 
                                        retain_graph=True, 
                                        create_graph=True)
        
        # Initialize a random vector
        grad_vector = torch.cat([g.reshape(-1) for g in gradients])
        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)

        vs = []
        idx = 0
        for g in gradients:
            sz = g.numel()
            vs.append(v[idx: idx + sz].view_as(g))
            idx += sz
        
        # Power iteration
        for _ in range(power_iter):
            dot = torch.dot(grad_vector, v)
            Hv_tuple = torch.autograd.grad(dot, weights, retain_graph=True)
            # Flatten the tuple of tensors into a single vector
            Hv = torch.cat([h.reshape(-1) for h in Hv_tuple])
            norm_Hv = torch.norm(Hv, p=2)
            if norm_Hv < 1e-8:
                return 0.0
            v = Hv / norm_Hv
        
        # Rayleigh quotient approximation for final eigenvalue
        Hv = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        lambda_max_approx = torch.dot(v, Hv).item()
        return lambda_max_approx
    
    def approximate_lambda_max(self, loss, model, power_iter=20):
        """
        Approximates the largest eigenvalue of the Hessian with respect to 'weights'
        using power iteration.
        Only works well if the Hessian is PSD.
        
        Returns: float lambda_max
        """
        # Compute first-order gradient with create_graph=True for higher-order derivatives.
        gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in gradients])
        
        # Initialize a random vector v of same shape as grad_vector
        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)
        
        for _ in range(power_iter):
            # Compute dot product between grad_vector and v (a scalar)
            dot = torch.dot(grad_vector, v)
            # Compute Hessian-vector product; allow_unused=True avoids error if some weights don't affect dot.
            Hv_tuple = torch.autograd.grad(dot, model.parameters(), retain_graph=True, allow_unused=True)
            # Replace any None gradients with zero tensors of the same shape
            Hv_list = []
            for idx, h in enumerate(Hv_tuple):
                Hv_list.append(h)
            # Flatten the gradients to a single vector
            Hv = torch.cat([h.reshape(-1) for h in Hv_list])
            norm_Hv = torch.norm(Hv, p=2)
            if norm_Hv < 1e-8:
                return 0.0
            v = Hv / norm_Hv

        # Final Rayleigh quotient approximation for the eigenvalue
        #final_dot = torch.dot(grad_vector, v)
        #lambda_max_approx = final_dot.item()
        lambda_max_approx = norm_Hv.item()
        return lambda_max_approx

    def compute_stable_rank(self, loss, batch_size, aggregate_method='mean'):
        """
        High-level function that:
        1) Computes the loss
        2) For each group:
           a) Approximates trace(H^2)
           b) Approximates largest eigenvalue
           c) stable_rank = trace(H^2) / (lambda_max^2)
        3) Stores results in self.stable_rank_per_group
        """
        # Compute the loss wrt all grouped weights (retain graph for second derivatives)
        
        stable_ranks = []
        lambda_max_list = []
        for weights in self.grouped_layer_weights:
            # 1. trace(H^2)
            trace_h2 = self.approximate_trace_h2(loss, weights)
            #print("method 1: ", trace_h2)
            # 2. largest eigenvalue (power iteration)
            lambda_max = self.approximate_lambda_max(loss, weights, power_iter=100)
            
            # 3. stable rank = trace(H^2) / (lambda_max^2)
            epsilon = 1e-12
            srank = trace_h2 / (lambda_max**2 + epsilon)
            
            stable_ranks.append(srank)
            lambda_max_list.append(lambda_max)
        
        #self.stable_rank_per_group = self.list_aggregate(self.stable_rank_per_group, stable_ranks, batch_size, aggregate_method)
        #self.lambda_max_list = self.list_aggregate(self.lambda_max_list, lambda_max_list, batch_size, aggregate_method)
        #print(self.stable_rank_per_group)
        return stable_ranks, lambda_max_list

    def compute_spectral_entropy(self, eigen_list, weight_list, sigma=0.01, grid=1000):
        # Step 1: Filter near-zero eigenvalues
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list)
        
        # Step 2: Renormalize weights
        renormalized_weight = renormalize_weights(filtered_weight)
        #print("renorm: ", sum(renormalized_weight))
        
        # Step 3: Define lambda grid
        if len(filtered_eigen) == 0:
            raise ValueError("No eigenvalues remain after filtering. Adjust the threshold.")
        lambda_min = min(filtered_eigen) - 1.0  # Adding padding
        lambda_max = max(filtered_eigen) + 1.0  # Adding padding
        lambdas = np.linspace(lambda_min, lambda_max, grid)
        delta_lambda = lambdas[1] - lambdas[0]
        
        # Step 4: Construct spectral density
        density = construct_spectral_density(filtered_eigen, renormalized_weight, lambdas, sigma)
        
        # Step 5: Compute spectral entropy
        epsilon = 1e-12
        p = density + epsilon  # Avoid log(0)
        #spectral_entropy = -np.sum(p * np.log(p))
        p = np.array(renormalized_weight) + epsilon
        spectral_entropy = -np.sum(p * np.log(p))
        weighted_entropy = -np.sum(p * np.log(p) * np.array(filtered_eigen))
        centroid = np.sum(np.array(renormalized_weight) * np.array(filtered_eigen)) 
        spread = np.sum(np.array(renormalized_weight) * (np.array(filtered_eigen) - centroid)**2)
        
        return spectral_entropy, weighted_entropy, centroid, spread

    def compute_effective_rank_old(self, eigen_list, weight_list):
        epsilon = 1e-12
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list, threshold=0)
        #print(filtered_eigen)
        weighted_eigen = np.array(filtered_eigen) * np.array(filtered_weight)
        #print(weighted_eigen)
        normalization = np.sum(weighted_eigen) + epsilon

        #print(normalization)
        p = weighted_eigen / (normalization + epsilon)
        p = np.array(p) + epsilon
        #print(p)
        entropy = -np.sum(p * np.log(p))

        effective_rank_entropy = np.exp(entropy)

        return effective_rank_entropy

    def compute_effective_rank(self, value_tensor: torch.Tensor, weight_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the effective rank (R_eff) using Option A: weighting by eigenvalue magnitude.
        
        Args:
            value_tensor (torch.Tensor): 1D tensor of eigenvalue bin centers.
            weight_tensor (torch.Tensor): 1D tensor of density values corresponding to the eigenvalues.
        
        Returns:
            torch.Tensor: Effective rank (a scalar tensor).
        """
        # Compute the bin width. We assume equally spaced bins.
        if value_tensor.numel() > 1:
            d_lambda = (value_tensor[1] - value_tensor[0]).item()
        else:
            d_lambda = 1.0  # Fallback value if only one bin is available
        
        # Compute the trace approximation T = sum(λ * density * bin_width)
        T = torch.sum(value_tensor * weight_tensor * d_lambda)
        
        # Construct the probability distribution:
        # p(λ) = (λ * density * bin_width) / T
        p = (value_tensor * weight_tensor * d_lambda) / T

        # For numerical stability, clamp p to avoid log(0)
        epsilon = 1e-12
        p_clamped = p.clamp(min=epsilon)
        
        # Compute the entropy H = -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p_clamped))
        
        # The effective rank is the exponential of the entropy.
        effective_rank = torch.exp(entropy)
        
        return effective_rank

    def spectral_density(self, n_iter=10, n_v=5, sigma=0.01, grid=100, threshold=1e-10):
        """
        Compute estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ). First compute the Hessian of all batches, then compute the values using the avearge Hessian.
        Parameters:
        -----------
        loss : torch.Tensor
            The loss tensor of the batch for which the Hessian is computed.
        batch_size : int
            The size of the batch.
        n_iter : int, optional (default=10)
            Number of iterations used to compute the trace.
        n_v : int, optional (default=5)
            Number of SLQ runs.
        sigma : float, optional (default=0.01)
            Standard deviation for Gaussian smoothing.
        grid : int, optional (default=100)
            Number of grid points for density estimation.
        threshold : float, optional (default=1e-10)
            Threshold for numerical stability.
        
        Saves:
        --------
        self.spectrum_divergence_list: 
            List of spectral divergences between the eigenvalue densities of each layer and the final layer.
        self.spectrum_entropy_list:
            List of spectral entropies of each layer.
        self.weighted_entropy_list:
            List of weighted entropies of each layer.
        self.centroid_list:
            List of centroids of each layer.
        self.spread_list:
            List of spreads of each layer.
        """

        self.sigma = sigma
        self.grid = grid
        self.threshold = threshold
        def group_product(xs, ys):
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
            #return [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]
        def group_add(params, update, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.add_(update[i] * alpha)
            return params
        device = self.device
        layer_eigenvalues = []
        layer_eigenweights = []
        layer_lambdas = []
        layer_density = []
        for weights in self.grouped_layer_weights:
            eigen_list_full, weight_list_full = [], []
            #gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            #print("grad shape: ", len(gradients))
            for k in range(n_v):
                v = [torch.randint_like(weight, high=2, device=device) for weight in weights]
                # generate Rademacher random variables
                for v_i in v:
                    v_i[v_i == 0] = -1
                v = normalization(v)

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                for i in range(n_iter):
                    self.model.zero_grad()
                    w_prime = [torch.zeros(weight.size()).to(device) for weight in weights]
                    if i == 0:
                        #w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        w_prime = self.dataloader_hv_product(v, weights)
            
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha)
                        w = group_add(w_prime, v, alpha=-alpha)
                        #print("w shape: ", len(w))
                        w_list.append(w)
                    else:
                        beta = torch.sqrt(group_product(w, w))
                        beta_list.append(beta.cpu().item())
                        if beta_list[-1] != 0.:
                            # We should re-orth it
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        else:
                            # generate a new vector
                            w = [torch.randn(weight.size()).to(device) for weight in weights]
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        #w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        w_prime = self.dataloader_hv_product(v, weights)
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha.cpu().item())
                        w_tmp = group_add(w_prime, v, alpha=-alpha)
                        w = group_add(w_tmp, v_list[-2], alpha=-beta)

                T = torch.zeros(n_iter, n_iter).to(device)
                for i in range(len(alpha_list)):
                    T[i, i] = alpha_list[i]
                    if i < len(alpha_list) - 1:
                        T[i + 1, i] = beta_list[i]
                        T[i, i + 1] = beta_list[i]
                a_, b_ = torch.linalg.eig(T)
                #print(a_)
                #print(b_)

                eigen_list = a_.real
                weight_list = b_[0, :].real**2
                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            
            layer_eigenvalues.append(flat_list(eigen_list_full))
            layer_eigenweights.append(flat_list(weight_list_full))

        spectrum_divergence_list = []
        for i in range(len(layer_eigenvalues)-1):
            divergence, _ = compute_spectral_divergences(layer_eigenvalues[i], layer_eigenweights[i], layer_eigenvalues[-1], layer_eigenweights[-1], measure='js')
            spectrum_divergence_list.append(divergence)


        #dis_1, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_2, _ = compute_spectral_divergences(layer_eigenvalues[1], layer_eigenweights[1], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_3, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[1], layer_eigenweights[1], measure='js')
        #print("dis: ", dis_1, dis_2, dis_3)
        #print(layer_lambdas[0])
        #print(layer_lambdas[1])
        #D_KL_test = kl_divergence(layer_density[0], layer_density[-1], layer_lambdas[0])
        #D_JS_test = js_divergence(density_test_1, density_test_2, lambdas_test)

        self.spectrum_entropy_list = []
        self.weighted_entropy_list = []
        self.centroid_list = []
        self.spread_list = []
        self.effective_rank_list = []
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            self.spectrum_entropy_list.append(spectral_entropy)
            self.weighted_entropy_list.append(weighted_entropy)
            self.centroid_list.append(centroid)
            self.spread_list.append(spread)
            effective_rank = self.compute_effective_rank(layer_eigenvalues[i], layer_eigenweights[i])
            self.effective_rank_list.append(effective_rank)

        self.spectrum_divergence_list = spectrum_divergence_list
        self.layer_eigenvalues = layer_eigenvalues
        self.layer_eigenweights = layer_eigenweights

        return layer_lambdas, layer_density
    
    def batch_spectral_density(self, loss, batch_size, n_iter=10, n_v=5, sigma=0.01, grid=100, threshold=1e-10):
        """
        Compute estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ). First compute the Hessian of the batch, then take the average over the batch in batch_aggregate.
        Parameters:
        -----------
        loss : torch.Tensor
            The loss tensor of the batch for which the Hessian is computed.
        batch_size : int
            The size of the batch.
        n_iter : int, optional (default=10)
            Number of iterations used to compute the trace.
        n_v : int, optional (default=5)
            Number of SLQ runs.
        sigma : float, optional (default=0.01)
            Standard deviation for Gaussian smoothing.
        grid : int, optional (default=100)
            Number of grid points for density estimation.
        threshold : float, optional (default=1e-10)
            Threshold for numerical stability.
        
        Saves:
        --------
        self.spectrum_divergence_list: 
            List of spectral divergences between the eigenvalue densities of each layer and the final layer.
        self.spectrum_entropy_list:
            List of spectral entropies of each layer.
        self.weighted_entropy_list:
            List of weighted entropies of each layer.
        self.centroid_list:
            List of centroids of each layer.
        self.spread_list:
            List of spreads of each layer.
        """
        self.sigma = sigma
        self.grid = grid
        self.threshold = threshold

        def group_product(xs, ys):
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
            #return [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]
        def group_add(params, update, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.add_(update[i] * alpha)
            return params
        def group_div(params, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.div_(alpha)
            return params
        def list_aggregate(list_a, list_b, batch_size):
            if len(list_a) == 0:
                list_a = [float(b) * batch_size for b in list_b]
            else:
                list_a = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
            return list_a
        device = self.device
        layer_eigenvalues = []
        layer_eigenweights = []
        layer_lambdas = []
        layer_density = []
        for weights in self.grouped_layer_weights:
            eigen_list_full, weight_list_full = [], []
            self.model.zero_grad()
            v = [torch.randn_like(weight, device=device) for weight in weights]
            v = torch.randn(sum([w.numel() for w in weights]))
            
            #n_param = sum([w.numel for w in weights])
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            #print(gradients[0].shape)
            #print("grad shape: ", len(gradients))
            for k in range(n_v):
                #v = [torch.randint_like(weight, high=2, device=device) for weight in weights]
                v = [torch.randn_like(weight, device=device) for weight in weights]

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                #for i in range(n_iter):
                self.model.zero_grad()
                alpha_list, beta_list = lanczos_gradient_single(self.model, loss, weights, n_iter)
                    
                alpha_tensor = torch.tensor(alpha_list, device=self.device)
                beta_tensor = torch.tensor(beta_list, device=self.device)

                T = torch.diag(alpha_tensor) + torch.diag(beta_tensor, 1) + torch.diag(beta_tensor, -1)
                a_, b_ = torch.linalg.eig(T)
                #print(a_)
                #print(b_)

                eigen_list = a_.real
                weight_list = b_[0, :].real**2
                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            
            layer_eigenvalues.append(flat_list(eigen_list_full))
            layer_eigenweights.append(flat_list(weight_list_full))

        spectrum_divergence_list = []
        for i in range(len(layer_eigenvalues)-1):
            divergence, _ = compute_spectral_divergences(layer_eigenvalues[i], layer_eigenweights[i], layer_eigenvalues[-1], layer_eigenweights[-1], measure='js')
            spectrum_divergence_list.append(divergence)
        #divergence, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-2], layer_eigenweights[-2], measure='js')
        #spectrum_divergence_list.append(divergence)
        #dis_1, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_2, _ = compute_spectral_divergences(layer_eigenvalues[1], layer_eigenweights[1], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_3, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[1], layer_eigenweights[1], measure='js')
        #print("dis: ", dis_1, dis_2, dis_3)
        #print(layer_lambdas[0])
        #print(layer_lambdas[1])
        #D_KL_test = kl_divergence(layer_density[0], layer_density[-1], layer_lambdas[0])
        #D_JS_test = js_divergence(density_test_1, density_test_2, lambdas_test)

        spectrum_entropy_list = []
        weighted_entropy_list = []
        centroid_list = []
        spread_list = []
        effective_rank_list = []
        stable_rank_list = []
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            spectrum_entropy_list.append(spectral_entropy * batch_size)
            weighted_entropy_list.append(weighted_entropy * batch_size)
            centroid_list.append(centroid * batch_size)
            spread_list.append(spread * batch_size)

            effective_rank = self.compute_effective_rank(layer_eigenvalues[i], layer_eigenweights[i])
            effective_rank_list.append(effective_rank * batch_size)

            #stable_rank = self.compute_stable_rank(layer_eigenvalues[i], layer_eigenweights[i])
            #stable_rank_list.append(stable_rank * batch_size)


        self.spectrum_entropy_list = list_aggregate(self.spectrum_entropy_list, spectrum_entropy_list, batch_size)
        self.weighted_entropy_list = list_aggregate(self.weighted_entropy_list, weighted_entropy_list, batch_size)
        self.centroid_list = list_aggregate(self.centroid_list, centroid_list, batch_size)
        self.spread_list = list_aggregate(self.spread_list, spread_list, batch_size)
        self.spectrum_divergence_list = list_aggregate(self.spectrum_divergence_list, spectrum_divergence_list, batch_size)

        self.effective_rank_list = list_aggregate(self.effective_rank_list, effective_rank_list, batch_size)

        #self.stable_rank_list = list_aggregate(self.stable_rank_list, stable_rank_list, batch_size)
        #print(len(self.spectrum_divergence_list))
        self.layer_eigenvalues = layer_eigenvalues * batch_size
        self.layer_eigenweights = layer_eigenweights * batch_size

        return layer_eigenvalues, layer_eigenweights

    def batch_spectral_density_old(self, loss, batch_size, n_iter=10, n_v=5, sigma=0.01, grid=100, threshold=1e-10):
        """
        Compute estimated eigenvalue density using the stochastic Lanczos algorithm (SLQ). First compute the Hessian of the batch, then take the average over the batch in batch_aggregate.
        Parameters:
        -----------
        loss : torch.Tensor
            The loss tensor of the batch for which the Hessian is computed.
        batch_size : int
            The size of the batch.
        n_iter : int, optional (default=10)
            Number of iterations used to compute the trace.
        n_v : int, optional (default=5)
            Number of SLQ runs.
        sigma : float, optional (default=0.01)
            Standard deviation for Gaussian smoothing.
        grid : int, optional (default=100)
            Number of grid points for density estimation.
        threshold : float, optional (default=1e-10)
            Threshold for numerical stability.
        
        Saves:
        --------
        self.spectrum_divergence_list: 
            List of spectral divergences between the eigenvalue densities of each layer and the final layer.
        self.spectrum_entropy_list:
            List of spectral entropies of each layer.
        self.weighted_entropy_list:
            List of weighted entropies of each layer.
        self.centroid_list:
            List of centroids of each layer.
        self.spread_list:
            List of spreads of each layer.
        """
        self.sigma = sigma
        self.grid = grid
        self.threshold = threshold

        def group_product(xs, ys):
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
            #return [torch.sum(x * y).cpu().item() for (x, y) in zip(xs, ys)]
        def group_add(params, update, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.add_(update[i] * alpha)
            return params
        def group_div(params, alpha=1):
            """
            params = params + update*alpha
            :param params: list of variable
            :param update: list of data
            :return:
            """
            for i, p in enumerate(params):
                params[i].data.div_(alpha)
            return params
        def list_aggregate(list_a, list_b, batch_size):
            if len(list_a) == 0:
                list_a = [float(b) * batch_size for b in list_b]
            else:
                list_a = [float(a + b * batch_size) for a, b in zip(list_a, list_b)]
            return list_a
        device = self.device
        layer_eigenvalues = []
        layer_eigenweights = []
        layer_lambdas = []
        layer_density = []
        for weights in self.grouped_layer_weights:
            eigen_list_full, weight_list_full = [], []
            self.model.zero_grad()
            gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
            #print("grad shape: ", len(gradients))
            for k in range(n_v):
                #v = [torch.randint_like(weight, high=2, device=device) for weight in weights]
                v = [torch.randn_like(weight, device=device) for weight in weights]
                # generate Rademacher random variables
                #for v_i in v:
                #    v_i[v_i == 0] = -1
                v = normalization(v)
                #v /= torch.norm(v)

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                for i in range(n_iter):
                    self.model.zero_grad()
                    w_prime = [torch.zeros(weight.size()).to(device) for weight in weights]
                    if i == 0:
                        w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        #w_prime = self.dataloader_hv_product(v, weights)
            
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha)
                        w = group_add(w_prime, v, alpha=-alpha)
                        #print("w shape: ", len(w))
                        w_list.append(w)
                    else:
                        beta = torch.sqrt(group_product(w, w))
                        beta_list.append(beta.cpu().item())
                        if beta_list[-1] != 0.:
                            # We should re-orth it
                            #v = orthnormal(w, v_list)
                            #v = w / beta_list[-1]
                            v = group_div(w, beta_list[-1])
                            v_list.append(v)
                        else:
                            # generate a new vector
                            w = [torch.randn(weight.size()).to(device) for weight in weights]
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        w_prime = torch.autograd.grad(gradients, weights, grad_outputs=v, only_inputs=True, retain_graph=True)
                        #w_prime = self.dataloader_hv_product(v, weights)
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha.cpu().item())
                        w_tmp = group_add(w_prime, v, alpha=-alpha)
                        w = group_add(w_tmp, v_list[-2], alpha=-beta)

                T = torch.zeros(n_iter, n_iter).to(device)
                for i in range(len(alpha_list)):
                    T[i, i] = alpha_list[i]
                    if i < len(alpha_list) - 1:
                        T[i + 1, i] = beta_list[i]
                        T[i, i + 1] = beta_list[i]
                a_, b_ = torch.linalg.eig(T)
                #print(a_)
                #print(b_)

                eigen_list = a_.real
                weight_list = b_[0, :].real**2
                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            
            layer_eigenvalues.append(flat_list(eigen_list_full))
            layer_eigenweights.append(flat_list(weight_list_full))

        spectrum_divergence_list = []
        for i in range(len(layer_eigenvalues)-1):
            divergence, _ = compute_spectral_divergences(layer_eigenvalues[i], layer_eigenweights[i], layer_eigenvalues[-1], layer_eigenweights[-1], measure='js')
            spectrum_divergence_list.append(divergence)
        #divergence, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-2], layer_eigenweights[-2], measure='js')
        #spectrum_divergence_list.append(divergence)
        #dis_1, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_2, _ = compute_spectral_divergences(layer_eigenvalues[1], layer_eigenweights[1], layer_eigenvalues[-1], layer_eigenweights[-1], measure='kl')
        #dis_3, _ = compute_spectral_divergences(layer_eigenvalues[0], layer_eigenweights[0], layer_eigenvalues[1], layer_eigenweights[1], measure='js')
        #print("dis: ", dis_1, dis_2, dis_3)
        #print(layer_lambdas[0])
        #print(layer_lambdas[1])
        #D_KL_test = kl_divergence(layer_density[0], layer_density[-1], layer_lambdas[0])
        #D_JS_test = js_divergence(density_test_1, density_test_2, lambdas_test)

        spectrum_entropy_list = []
        weighted_entropy_list = []
        centroid_list = []
        spread_list = []
        effective_rank_list = []
        stable_rank_list = []
        for i in range(len(layer_eigenvalues)):
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(layer_eigenvalues[i], layer_eigenweights[i], sigma=self.sigma, grid=self.grid)
            #print(spectral_entropy)
            spectrum_entropy_list.append(spectral_entropy * batch_size)
            weighted_entropy_list.append(weighted_entropy * batch_size)
            centroid_list.append(centroid * batch_size)
            spread_list.append(spread * batch_size)

            effective_rank = self.compute_effective_rank(layer_eigenvalues[i], layer_eigenweights[i])
            effective_rank_list.append(effective_rank * batch_size)

            #stable_rank = self.compute_stable_rank(layer_eigenvalues[i], layer_eigenweights[i])
            #stable_rank_list.append(stable_rank * batch_size)


        self.spectrum_entropy_list = list_aggregate(self.spectrum_entropy_list, spectrum_entropy_list, batch_size)
        self.weighted_entropy_list = list_aggregate(self.weighted_entropy_list, weighted_entropy_list, batch_size)
        self.centroid_list = list_aggregate(self.centroid_list, centroid_list, batch_size)
        self.spread_list = list_aggregate(self.spread_list, spread_list, batch_size)
        self.spectrum_divergence_list = list_aggregate(self.spectrum_divergence_list, spectrum_divergence_list, batch_size)

        self.effective_rank_list = list_aggregate(self.effective_rank_list, effective_rank_list, batch_size)

        #self.stable_rank_list = list_aggregate(self.stable_rank_list, stable_rank_list, batch_size)
        #print(len(self.spectrum_divergence_list))
        self.layer_eigenvalues = layer_eigenvalues * batch_size
        self.layer_eigenweights = layer_eigenweights * batch_size

        return layer_eigenvalues, layer_eigenweights

    # From 'Why TF needs Adam'
    def get_layer_spectrum(self, n_v, n_iter):
        weights_dic, values_dic = {}, {}

        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:

                zeros = np.zeros((n_v, n_iter))
                weights_dic[name] = [row.tolist() for row in zeros]
                values_dic[name] =  [row.tolist() for row in zeros]

    
        t_s = time.time()
        for k in range(n_v): 
            #print('current k' , k)

            'wiki version'
            T_dic = self.tridiagonalize_by_lanzcos_layer_by_layer(n_iter, k) #returns a dic: {'name': T}
            
            for name, T in T_dic.items():
                eigenvalues, U  = np.linalg.eigh(T)
                values_dic[name][k] = eigenvalues.tolist() #array to list
                weights_dic[name][k] = (U[0]**2).tolist()

            #print("===values: ", eigenvalues)
            #'we also save the inter-medium results'
            #self.save_curve(total_time= time.time() - t_s, weights_layer = weights_dic, values_layer = values_dic)
        #print(weights_dic.keys())
        for name in values_dic.keys():
            values_dic[name] = np.concatenate(values_dic[name])
            weights_dic[name] = np.concatenate(weights_dic[name])

        return values_dic, weights_dic

        total_time = time.time() - t_s

        self.save_curve(total_time= total_time, weights_layer = weights_dic, values_layer = values_dic)

    def tridiagonalize_by_lanzcos_layer_by_layer(self, n_iter, k):
        v_dic = {} # value: list
        alpha_dic = {} # value: scaler
        w_dic = {} # value: #parameters*1 tensor
        beta_dic = {} # value: scaler
        T_dic = {} # value: m*m tensor 
        'initialize'
        for name, params in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if params.requires_grad:
                v = torch.randn_like(params, dtype = torch.float64) 
                v /= torch.norm(v)
                v_dic[name] = [v.cpu()]
                T_dic[name] = np.zeros((n_iter, n_iter), dtype= np.float64)
        #print(v_dic)

        w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,0) 

        'orthogonalize wprime'
        for name in T_dic.keys():
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name] 

        'iteration'
        for j in range(1, n_iter):

            for name in T_dic.keys(): 
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta >1e-8:
                    v_dic[name].append( w_dic[name] / beta )
                else:
                    #print('The value of beta is 0')
                    v_dic[name].append( w_dic[name] / 1e-8 )
                    #raise ZeroDivisionError('The value of beta is 0')
                if len(v_dic[name]) > 2:
                    del v_dic[name][0]  # keep this list short to save memory

            t_hessian = time.time()
  
            w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,j) 

            'orthogonalize wprime'
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name] 
                T_dic[name][j-1, j ] = beta_dic[name] 
                T_dic[name][j , j-1] = beta_dic[name]

        return  T_dic

    def get_full_spectrum(self, n_v, n_iter, dataloader=None):
        weights = np.zeros((n_v, n_iter))
        values = np.zeros((n_v, n_iter))

        for k in range(n_v): 
            'wiki version'
            T = self.tridiagonalize_by_lanzcos(n_iter, k, dataloader)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
        
        all_values = np.concatenate(values)
        all_weights = np.concatenate(weights)
        return all_values, all_weights
   
        grid, curve = self.interpolate(weights, values)
    
    def tridiagonalize_by_lanzcos(self, n_iter, k, dataloader=None):
        'set up'
        v_list = []
        T = np.zeros((n_iter, n_iter), dtype= np.float64)

        'initialization'
        v = torch.randn(self.total_params, dtype = torch.float64) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], dataloader)
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        #print('runing lanczos')
        for j in range(1, n_iter):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory


            w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], dataloader)
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta
         
        return  T

    def hessian_vector_product_with_tensor_input(self, d_tensor, dataloader=None):
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'
        #if dataloader is None:
        #    dataloader = self.dataloader
        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch in dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            loss = self.loss_fn(output, target, 'mean')
            #self.model.zero_grad()

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor * batch_size

        total_hd_tensor /= len(dataloader.dataset)
            #if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
            #    break
        return total_hd_tensor

    def hessian_vector_product_with_dic_input(self, d_dic, dataloader=None):
        'comput hessian_vector product, takes a dictionary as input, the values of dic is a list of historical lanscoz directions: d_dic = {name, [history v..]}'
        if dataloader is None:
            dataloader = self.dataloader
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        'initialize'
        hd_dic = {}
        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:
                hd_dic[name]  = torch.zeros_like(param.data).cpu()


        t_hd = time.time()
        for batch in dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            loss = self.loss_fn(output, target, 'mean')
            loss.backward(create_graph= True)
            g_dic = {}
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    g_dic[name] = param.grad.double()

        
            self.model.zero_grad(set_to_none = True)
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    l = torch.sum(g_dic[name].cuda() * d_dic[name][-1].cuda())
                    l.backward(retain_graph = True)
                    hd = param.grad.double().data.clone()
                    hd_dic[name]  += hd.cpu()   
                    self.model.zero_grad(set_to_none = True)
            break
       
        return hd_dic

    # Compute the spectrum only
    def compute_spectrum(self, train_num, valid_num, n_iter=10, n_v=5, method=1):
        with sdpa_kernel(SDPBackend.MATH):
            if method == 1:
                # First compute the Hessian for all batches. Then compute the spectral density
                self.spectral_density(n_iter=n_iter, n_v=n_v, sigma=0.01)
            elif method == 2:
                # First compute the Hessian and spectral density for each batch. Then aggregate the avearge for all batches.
                device = self.device
                model = self.model
                loss_fn = self.loss_fn
                for batch in self.dataloader:
                    data, target, batch_size = self.load_batch_func(batch, device)
                    output = model(data)
                    loss = loss_fn(output, target, 'none')
                    model.zero_grad()

                    self.batch_spectral_density(loss.mean(), batch_size, n_iter=n_iter, n_v=n_v)

                self.spectrum_divergence_list = self.group_div_const(self.spectrum_divergence_list, train_num)
                self.centroid_list = self.group_div_const(self.centroid_list, train_num)
                self.spread_list = self.group_div_const(self.spread_list, train_num)
                self.weighted_entropy_list = self.group_div_const(self.weighted_entropy_list, train_num)
                self.spectrum_entropy_list = self.group_div_const(self.spectrum_entropy_list, train_num)
                self.effective_rank_list = self.group_div_const(self.effective_rank_list, train_num)
                self.stable_rank_list = self.group_div_const(self.stable_rank_list, train_num)
            else:
                print("=======> SLQ for full model")
                values_full, weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter)
                self.values_full = values_full.tolist()
                self.weights_full = weights_full.tolist()
                spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(values_full, weights_full, sigma=0.01, grid=1000)
                self.spectrum_entropy_list.append(spectral_entropy)
                self.weighted_entropy_list.append(weighted_entropy)
                self.centroid_list.append(centroid)
                self.spread_list.append(spread)
                effective_rank = self.compute_effective_rank(values_full, weights_full)
                self.effective_rank_list.append(effective_rank)
                filtered_eigens, _ = filter_eigenvalues(values_full, weights_full)
                self.condition_list.append(np.abs(np.max(filtered_eigens)) / np.abs(np.min(filtered_eigens)))
                #print(values_full)
                
                """
                print("SLQ for layers")
                values_dic, weights_dic = self.get_layer_spectrum(n_v=n_v, n_iter=n_iter)
                self.values_head = values_dic['head.weight'].tolist()
                self.weights_head = weights_dic['head.weight'].tolist()

                for name in values_dic.keys():
                    spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(values_dic[name], weights_dic[name], sigma=0.01, grid=1000)
                    self.spectrum_entropy_list.append(spectral_entropy)
                    self.weighted_entropy_list.append(weighted_entropy)
                    self.centroid_list.append(centroid)
                    self.spread_list.append(spread)
                    effective_rank = self.compute_effective_rank(values_dic[name], weights_dic[name])
                    self.effective_rank_list.append(effective_rank)
                    filtered_eigens, _ = filter_eigenvalues(values_dic[name], weights_dic[name])
                    self.condition_list.append(np.abs(np.max(filtered_eigens)) / np.abs(np.min(filtered_eigens)))

                for name in values_dic.keys():
                    values_dic[name] = values_dic[name].tolist()
                    weights_dic[name] = weights_dic[name].tolist()
                self.values_dic = values_dic
                self.weights_dic = weights_dic
                """

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None   

    def collect2(self, logger, log_i, train_num, valid_num, n_iter=10, n_v=5):
        # compute hessian trace
        print("=======> Hessian trace")
        train_hessian, train_hessian_2, valid_hessian, valid_hessian_2 = self.compare_hessian(logger, log_i, train_num, valid_num)

        print("=======> SLQ for full model")
        with sdpa_kernel(SDPBackend.MATH):
            values_full, weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter, dataloader=self.dataloader)
            self.values_full = values_full.tolist()
            self.weights_full = weights_full.tolist()
            train_lambda_max = np.max(values_full)
            train_stable_rank = train_hessian_2 / train_lambda_max
            train_lambda_max_ratio = train_lambda_max / train_hessian
            spectral_entropy, weighted_entropy, centroid, spread = self.compute_spectral_entropy(values_full, weights_full, sigma=0.01, grid=1000)
            effective_rank = self.compute_effective_rank(values_full, weights_full)
            filtered_eigens, _ = filter_eigenvalues(values_full, weights_full)
            condition = np.abs(np.max(filtered_eigens)) / np.abs(np.min(filtered_eigens))

            values_tensor = torch.tensor(values_full, dtype=torch.float32)
            weights_tensor = torch.tensor(weights_full, dtype=torch.float32)
            pos_mask = values_tensor > 0
            neg_mask = values_tensor < 0
            d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            train_neg_trace = (-torch.sum(values_tensor[neg_mask] * weights_tensor[neg_mask]) * d).item()
            train_pos_trace = (torch.sum(values_tensor[pos_mask] * weights_tensor[pos_mask]) * d).item()

            # Compute metrics.
            train_tmf = tail_mass_fraction(values_tensor, weights_tensor, quantile=0.9)
            train_tmf_pos = tail_mass_fraction(values_tensor[pos_mask], weights_tensor[pos_mask], quantile=0.9)
            train_tmf_neg = tail_mass_fraction(-values_tensor[neg_mask], weights_tensor[neg_mask], quantile=0.9)
            train_gini = weighted_gini(values_tensor, weights_tensor)
            train_gini_pos = weighted_gini(values_tensor[pos_mask], weights_tensor[pos_mask])
            train_gini_neg = weighted_gini(-values_tensor[neg_mask], weights_tensor[neg_mask])
            train_skew = weighted_skewness(values_tensor, weights_tensor)
            train_skew_pos = weighted_skewness(values_tensor[pos_mask], weights_tensor[pos_mask])
            train_skew_neg = weighted_skewness(-values_tensor[neg_mask], weights_tensor[neg_mask])

        logger.log("train_values_full", self.values_full, log_i)
        logger.log("train_weights_full", self.weights_full, log_i)
        logger.log("train_entropy", spectral_entropy, log_i)
        logger.log("train_weighted_entropy", weighted_entropy, log_i)
        logger.log("train_centroid", centroid, log_i)
        logger.log("train_spread", spread, log_i)
        logger.log("train_effective_rank", effective_rank, log_i)
        logger.log("train_stable_rank", train_stable_rank, log_i)
        logger.log("train_condition", condition, log_i)

        logger.log("train_pos_trace", train_pos_trace, log_i)
        logger.log("train_neg_trace", train_neg_trace, log_i)
        logger.log("train_lambda_max", train_lambda_max, log_i)
        logger.log("train_lambda_max_ratio", train_lambda_max_ratio, log_i)

        logger.log("train_tmf", train_tmf, log_i)
        logger.log("train_tmf_pos", train_tmf_pos, log_i)
        logger.log("train_tmf_neg", train_tmf_neg, log_i)
        logger.log("train_gini", train_gini, log_i)
        logger.log("train_gini_pos", train_gini_pos, log_i)
        logger.log("train_gini_neg", train_gini_neg, log_i)
        logger.log("train_gini_minus", train_gini_pos - train_gini_neg, log_i)
        logger.log("train_skew", train_skew, log_i)
        logger.log("train_skew_pos", train_skew_pos, log_i)
        logger.log("train_skew_neg", train_skew_neg, log_i)

        with sdpa_kernel(SDPBackend.MATH):
            valid_values_full, valid_weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter, dataloader=self.valid_dataloader)
            self.valid_values_full = valid_values_full.tolist()
            self.valid_weights_full = valid_weights_full.tolist()
            valid_lambda_max = np.max(valid_values_full)
            valid_stable_rank = valid_hessian_2 / valid_lambda_max
            valid_lambda_max_ratio = valid_lambda_max / valid_hessian
            valid_spectral_entropy, valid_weighted_entropy, valid_centroid, valid_spread = self.compute_spectral_entropy(valid_values_full, valid_weights_full, sigma=0.01, grid=1000)
            valid_effective_rank = self.compute_effective_rank(valid_values_full, valid_weights_full)
            valid_filtered_eigens, _ = filter_eigenvalues(valid_values_full, valid_weights_full)
            valid_condition = np.abs(np.max(valid_filtered_eigens)) / np.abs(np.min(valid_filtered_eigens))

            valid_values_tensor = torch.tensor(valid_values_full, dtype=torch.float32)
            valid_weights_tensor = torch.tensor(valid_weights_full, dtype=torch.float32)
            valid_pos_mask = valid_values_tensor > 0
            valid_neg_mask = valid_values_tensor < 0
            d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            valid_neg_trace = (-torch.sum(valid_values_tensor[valid_neg_mask] * valid_weights_tensor[valid_neg_mask]) * d).item()
            valid_pos_trace = (torch.sum(valid_values_tensor[valid_pos_mask] * valid_weights_tensor[valid_pos_mask]) * d).item()

            # Compute metrics.
            valid_tmf = tail_mass_fraction(valid_values_tensor, valid_weights_tensor, quantile=0.9)
            valid_tmf_pos = tail_mass_fraction(valid_values_tensor[valid_pos_mask], valid_weights_tensor[valid_pos_mask], quantile=0.9)
            valid_tmf_neg = tail_mass_fraction(-valid_values_tensor[valid_neg_mask], valid_weights_tensor[valid_neg_mask], quantile=0.9)
            valid_gini = weighted_gini(valid_values_tensor, valid_weights_tensor)
            valid_gini_pos = weighted_gini(valid_values_tensor[valid_pos_mask], valid_weights_tensor[valid_pos_mask])
            valid_gini_neg = weighted_gini(-valid_values_tensor[valid_neg_mask], valid_weights_tensor[valid_neg_mask])
            valid_skew = weighted_skewness(valid_values_tensor, valid_weights_tensor)
            valid_skew_pos = weighted_skewness(valid_values_tensor[valid_pos_mask], valid_weights_tensor[valid_pos_mask])
            valid_skew_neg = weighted_skewness(-valid_values_tensor[valid_neg_mask], valid_weights_tensor[valid_neg_mask])

        logger.log("valid_values_full", self.valid_values_full, log_i)
        logger.log("valid_weights_full", self.valid_weights_full, log_i)
        logger.log("valid_entropy", valid_spectral_entropy, log_i)
        logger.log("valid_weighted_entropy", valid_weighted_entropy, log_i)
        logger.log("valid_centroid", valid_centroid, log_i)
        logger.log("valid_spread", valid_spread, log_i)
        logger.log("valid_effective_rank", valid_effective_rank, log_i)
        logger.log("valid_stable_rank", valid_stable_rank, log_i)
        logger.log("valid_condition", valid_condition, log_i)

        logger.log("valid_pos_trace", valid_pos_trace, log_i)
        logger.log("valid_neg_trace", valid_neg_trace, log_i)
        logger.log("valid_lambda_max", valid_lambda_max, log_i)
        logger.log("valid_lambda_max_ratio", valid_lambda_max_ratio, log_i)

        logger.log("valid_tmf", valid_tmf, log_i)
        logger.log("valid_tmf_pos", valid_tmf_pos, log_i)
        logger.log("valid_tmf_neg", valid_tmf_neg, log_i)
        logger.log("valid_gini", valid_gini, log_i)
        logger.log("valid_gini_pos", valid_gini_pos, log_i)
        logger.log("valid_gini_neg", valid_gini_neg, log_i)
        logger.log("valid_gini_minus", valid_gini_pos - valid_gini_neg, log_i)
        logger.log("valid_skew", valid_skew, log_i)
        logger.log("valid_skew_pos", valid_skew_pos, log_i)
        logger.log("valid_skew_neg", valid_skew_neg, log_i)

        plot_curves(log=logger, data_names=["train_entropy", "valid_entropy"], path_name='entropy', file_name='entropy')
        plot_curves(log=logger, data_names=["train_weighted_entropy", "valid_weighted_entropy"], path_name='entropy', file_name='weighted_entropy')
        plot_curves(log=logger, data_names=["train_centroid", "valid_centroid"], path_name='entropy', file_name='centroid')
        plot_curves(log=logger, data_names=["train_spread", "valid_spread"], path_name='entropy', file_name='spread')
        plot_curves(log=logger, data_names=["train_effective_rank", "valid_effective_rank"], path_name='entropy', file_name='effective_rank')
        plot_curves(log=logger, data_names=["train_stable_rank", "valid_stable_rank"], path_name='entropy', file_name='stable_rank')
        plot_curves(log=logger, data_names=["train_condition", "valid_condition"], path_name='entropy', file_name='condition')

        plot_curves(log=logger, data_names=["train_pos_trace", "valid_pos_trace"], path_name='hessian', file_name='pos_trace')
        plot_curves(log=logger, data_names=["train_neg_trace", "valid_neg_trace"], path_name='hessian', file_name='neg_trace')

        plot_curves(log=logger, data_names=["train_tmf", "valid_tmf"], path_name='distribution', file_name='tmf')
        plot_curves(log=logger, data_names=["train_tmf_pos", "valid_tmf_pos"], path_name='distribution', file_name='tmf_pos')
        plot_curves(log=logger, data_names=["train_tmf_neg", "valid_tmf_neg"], path_name='distribution', file_name='tmf_neg')
        plot_curves(log=logger, data_names=["train_gini", "valid_gini"], path_name='distribution', file_name='gini', y_log=False)
        plot_curves(log=logger, data_names=["train_gini_pos", "valid_gini_pos"], path_name='distribution', file_name='gini_pos', y_log=False)
        plot_curves(log=logger, data_names=["train_gini_neg", "valid_gini_neg"], path_name='distribution', file_name='gini_neg', y_log=False)
        plot_curves(log=logger, data_names=["train_skew", "valid_skew"], path_name='distribution', file_name='skew')
        plot_curves(log=logger, data_names=["train_skew_pos", "valid_skew_pos"], path_name='distribution', file_name='skew_pos')
        plot_curves(log=logger, data_names=["train_skew_neg", "valid_skew_neg"], path_name='distribution', file_name='skew_neg')

        # landscape
        plot_curves(log=logger, data_names=["train_hessian", "valid_hessian"], path_name='landscape', file_name='hessian')
        plot_curves(log=logger, data_names=["train_lambda_max", "valid_lambda_max"], path_name='landscape', file_name='lambda_max')
        plot_curves(log=logger, data_names=["train_lambda_max_ratio", "valid_lambda_max_ratio"], path_name='landscape', file_name='lambda_max_ratio')
        plot_curves(log=logger, data_names=["train_condition", "valid_condition"], path_name='landscape', file_name='condition')
        plot_curves(log=logger, data_names=["train_stable_rank", "valid_stable_rank"], path_name='landscape', file_name='stable_rank')
        plot_curves(log=logger, data_names=["train_gini_pos", "valid_gini_pos"], path_name='landscape', file_name='gini_pos', y_log=False)
        #plot_curves(log=logger, data_names=["train_gini_neg", "valid_gini_neg"], path_name='landscape', file_name='gini_neg', y_log=False)
        plot_curves(log=logger, data_names=["train_gini_minus", "valid_gini_minus"], path_name='landscape', file_name='gini_minus', y_log=False)


    def pac_bound(self, logger, log_i, train_num, valid_num, n_iter=10, n_v=5, delta = 0.1, use_hessian_bound=True):
        """
        print("=======> lambda")
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            
            train_lambda_max = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                #model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(loss.mean(), model, power_iter=100)
                train_lambda_max += batch_lambda_max * batch_size
            train_lambda_max /= len(self.dataloader.dataset)

            valid_lambda_max = 0
            for batch in self.valid_dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                #model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(loss.mean(), model, power_iter=100)
                valid_lambda_max += batch_lambda_max * batch_size
            valid_lambda_max /= len(self.valid_dataloader.dataset)
        """
        print("=======> hessian")
        train_hessian, train_hessian_2, valid_hessian, valid_hessian_2 = self.compare_hessian(logger, log_i, train_num, valid_num)
        d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        delta = torch.tensor(delta)

        print("part 1", 2 * torch.sqrt(train_hessian_2*torch.log(2*d/delta)) / torch.sqrt(torch.tensor(train_num)))
        print("part 2", 4 * train_hessian * torch.log(2/delta) / (3 * train_num))
        #hessian_part = 2 * torch.sqrt(train_hessian_2*torch.log(2*d/delta)) / torch.sqrt(torch.tensor(train_num)) + 4 * train_hessian * torch.log(2/delta) / (3 * train_num)
        hessian_part = torch.sqrt(2*train_hessian_2*torch.log(d/delta) / train_num) + (train_hessian+torch.sqrt(torch.tensor(train_hessian_2)))*torch.log(2/delta) / (3*train_num)

        hessian_item = abs(train_hessian - valid_hessian)
        
        kl_div = compute_kl_divergence_initial_state(self.model.state_dict(), self.model.init_state)
        pac_part = pac_bayes_term(kl_div=kl_div, n=train_num, delta=delta)

        pac_part = pac_part.item()
        hessian_part = hessian_part.item()
        result = pac_part + hessian_part
        real_result = pac_part + hessian_item.item()
        logger.log("pac_part", pac_part, log_i)
        logger.log("hessian_part", hessian_part, log_i)
        logger.log("pac_bound_result", result, log_i)
        logger.log("pac_real_result", real_result, log_i)
        plot_curves(log=logger, data_names=["pac_part", "hessian_part"], path_name='pac', file_name='pac')

        logger.log("hessian_gap", abs(train_hessian - valid_hessian), log_i)
        plot_curves(log=logger, data_names=["hessian_gap", "hessian_part"], path_name='hessian_noise', file_name='hessian_gap')

        plot_curves(log=logger, data_names=['pac_bound_result', 'pac_real_result', 'loss_gap'], path_name='gap', file_name='pac_bound')

        return result, pac_part, hessian_part, real_result
    
    def noisy_loss(self, logger, log_i, train_loss, val_loss, train_num, valid_num):
        print("=======> hessian")
        train_hessian, train_hessian_2, valid_hessian, valid_hessian_2 = self.compare_hessian(logger, log_i, train_num, valid_num)
        print("=======> perturb model")
        noisy_model = copy.deepcopy(self.model)
        loss_fn = self.loss_fn

        noisy_train_loss_list = []
        noisy_valid_loss_list = []
        sigma = 0.01
        for i in range(50):
            for param in noisy_model.parameters():
                noise = torch.randn_like(param) * sigma
                param.data.add_(noise)

            noisy_train_loss = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, self.device)
                output = noisy_model(data)
                loss = self.loss_fn(output, target, 'mean')
                noisy_train_loss += loss.item() * batch_size
            noisy_train_loss /= train_num
            noisy_train_loss_list.append(noisy_train_loss)

            noisy_valid_loss = 0
            for batch in self.valid_dataloader:
                data, target, batch_size = self.load_batch_func(batch, self.device)
                output = noisy_model(data)
                loss = self.loss_fn(output, target, 'mean')
                noisy_valid_loss += loss.item() * batch_size
            noisy_valid_loss /= valid_num
            noisy_valid_loss_list.append(noisy_valid_loss)

        noisy_train_loss = np.mean(noisy_train_loss_list)
        noisy_valid_loss = np.mean(noisy_valid_loss_list)
        perturbation_loss_gap = noisy_valid_loss - noisy_train_loss
        logger.log("perturbation_loss_gap", perturbation_loss_gap, log_i)
        logger.log("noisy_train_loss", noisy_train_loss, log_i)
        logger.log("noisy_valid_loss", noisy_valid_loss, log_i)

        loss_gap = abs(train_loss - val_loss)
        hessian_gap = train_hessian - valid_hessian
        #perturbation_taylor_gap = loss_gap + (valid_hessian - train_hessian) * sigma**2 / 2
        perturbation_taylor_gap = loss_gap + (train_hessian - valid_hessian) * sigma**2 / 2
        train_taylor_loss = train_loss + train_hessian * sigma**2 / 2
        valid_taylor_loss = val_loss + valid_hessian * sigma**2 / 2
        logger.log("perturbation_taylor_gap", perturbation_taylor_gap, log_i)
        logger.log("train_taylor_loss", train_taylor_loss, log_i)
        logger.log("valid_taylor_loss", valid_taylor_loss, log_i)
        logger.log("hessian_gap", hessian_gap, log_i)
        plot_curves(log=logger, data_names=["noisy_train_loss", "train_taylor_loss"], path_name='perturbation', file_name='train')
        plot_curves(log=logger, data_names=["noisy_valid_loss", "valid_taylor_loss"], path_name='perturbation', file_name='valid')
        plot_curves(log=logger, data_names=['perturbation_loss_gap', 'perturbation_taylor_gap'], path_name='perturbation', file_name='gap')
        plot_curves(log=logger, data_names=['hessian_gap'], path_name='perturbation', file_name='hessian_gap', y_log=False)

    def compute_compression_bound(self, logger, log_i, train_num, valid_num, train_loss):
        #vector = self.model.weight.cpu().data.numpy()
        vector = torch.cat([p.detach().view(-1) for p in self.model.parameters()]).cpu().numpy()
        quantized_vec, message_len = quantize_vector(vector)
        prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
        misc_extra_bits = 5 #TODO
        divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
        total_sample_size = train_num + valid_num
        sample_size = train_num
        alpha = 0.2
        all_selected_probs = []
        for batch in self.valid_dataloader:
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            #loss = self.loss_fn(output, target, 'mean')
            logits = output[:, -1, :]
            softmax_matrix = torch.nn.functional.softmax(logits,dim=-1)
            selected_prob_scores = softmax_matrix[torch.arange(softmax_matrix.shape[0]), target[:, -1].view(-1)]
            all_selected_probs.append(selected_prob_scores)
        all_selected_probs = torch.cat(all_selected_probs)

        vocab_size = self.p
        log_probs = [np.log2((1-alpha)*x.item() + alpha/vocab_size) for x in selected_prob_scores]
        bdp_alpha = - sum(log_probs) / len(log_probs)
        alpha = (bdp_alpha) / (valid_num) # TODO

        delta = np.log2(1 + (1 - alpha) * vocab_size / alpha)
        compression_bound = llm_subsampling_bound(train_error=train_loss, div=divergence, data_size=total_sample_size, sample_size=sample_size, delta=delta)
        compression_bound = compression_bound - train_loss
        logger.log("compression_bound", compression_bound, log_i)
        plot_curves(log=logger, data_names=['loss_gap', 'compression_bound'], path_name='compression', file_name='compression_bound')
        return compression_bound

    def compare_bound(self, logger, log_i, train_num, valid_num, train_loss, n_iter=10, n_v=5):
        # compte compression bound baseline
        compression_bound = self.compute_compression_bound(logger, log_i, train_num, valid_num, train_loss)
        
        # approximate Hessian spectrum using SLQ, on training data and validation data
        print("=======> SLQ")
        with sdpa_kernel(SDPBackend.MATH):
            train_values_full, train_weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter, dataloader=self.dataloader)
            valid_values_full, valid_weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter, dataloader=self.valid_dataloader)

        # model norm, use in the Hessian bound
        r = compute_model_norm(self.model)

        train_values_tensor = torch.tensor(train_values_full, dtype=torch.float32)
        train_weights_tensor = torch.tensor(train_weights_full, dtype=torch.float32)
        train_pos_mask = train_values_tensor > 0
        train_neg_mask = train_values_tensor < 0

        valid_values_tensor = torch.tensor(valid_values_full, dtype=torch.float32)
        valid_weights_tensor = torch.tensor(valid_weights_full, dtype=torch.float32)
        valid_pos_mask = valid_values_tensor > 0
        valid_neg_mask = valid_values_tensor < 0

        # Some spectrum metrics that might be useful: effective rank , entropy, weighted entropy
        train_effective_rank = self.compute_effective_rank(train_values_tensor[train_pos_mask], train_weights_tensor[train_pos_mask])
        train_entropy, train_weighted_entropy, train_centroid, train_spread = self.compute_spectral_entropy(train_values_full[train_pos_mask], train_weights_full[train_pos_mask], sigma=0.01, grid=1000)
        valid_effective_rank = self.compute_effective_rank(valid_values_tensor[valid_pos_mask], valid_weights_tensor[valid_pos_mask])
        valid_entropy, valid_weighted_entropy, valid_centroid, valid_spread = self.compute_spectral_entropy(valid_values_full[valid_pos_mask], valid_weights_full[valid_pos_mask], sigma=0.01, grid=1000)

        # Compute the trace bound and spectral bound
        d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        C = np.sqrt(5)
        
        # trace and bound, computed on validation data
        valid_hessian_trace = torch.sum(valid_values_tensor * valid_weights_tensor) * d
        trace_bound = torch.sqrt(torch.tensor(valid_hessian_trace/valid_num)) * r
        spectral_bound = torch.sqrt(torch.tensor(valid_weighted_entropy/valid_num)) * r

        # log results to logger
        logger.log("train_effective_rank", train_effective_rank.item(), log_i)
        logger.log("train_entropy", train_entropy.item(), log_i)
        logger.log("train_weighted_entropy", train_weighted_entropy.item(), log_i)
        logger.log("valid_effective_rank", valid_effective_rank.item(), log_i)
        logger.log("valid_entropy", valid_entropy.item(), log_i)
        logger.log("valid_weighted_entropy", valid_weighted_entropy.item(), log_i)
        logger.log("trace_bound", trace_bound.item(), log_i)
        logger.log("spectral_bound", spectral_bound.item(), log_i)

        # plot results in logger
        plot_curves(log=logger, data_names=['train_effective_rank', 'valid_effective_rank'], path_name='entropy', file_name='effective_rank')
        plot_curves(log=logger, data_names=['train_entropy', 'valid_entropy'], path_name='entropy', file_name='entropy')
        plot_curves(log=logger, data_names=['train_weighted_entropy', 'valid_weighted_entropy'], path_name='entropy', file_name='weighted_entropy')
        plot_curves(log=logger, data_names=["loss_gap", "trace_bound", "spectral_bound", "compression_bound"], path_name='bound', file_name='bound')


    def log(self, logger, i):
        
        logger.log("values_full", self.values_full, i)
        logger.log("weights_full", self.weights_full, i)
        #logger.log("values_head", self.values_head, i)
        #logger.log("weights_head", self.weights_head, i)
        #logger.log("values_dic", self.values_dic, i)
        #logger.log("weights_dic", self.weights_dic, i)
        logger.log("spectral_entropy", self.spectrum_entropy_list, i)
        logger.log("weighted_entropy", self.weighted_entropy_list, i)
        logger.log("centroid", self.centroid_list, i)
        logger.log("spread", self.spread_list, i)
        #logger.log("spectrum_divergence", self.spectrum_divergence_list, i)
        logger.log("effective_rank", self.effective_rank_list, i)
        #logger.log("shapescale", self.shapescale, i)
        #logger.log("stable_rank", self.stable_rank_list, i)
        logger.log("condition", self.condition_list, i)
        
        logger.log("train_hessian", self.train_hessian, i)
        logger.log("valid_hessian", self.valid_hessian, i)


def compute_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        
        eigenvalues = None
        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])

        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)
        
        # Compute the dot product between gradients and noise vector.
        grad_dot_noise = torch.dot(grad_vector, v)
        
        # Compute Hessian-vector product using the Pearlmutter trick.
        Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
            tmp_eigenvalues = torch.sum(Hv*v).cpu().item()

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors

def compute_layer_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    layers = model.get_layers()
    weights = [module.weight for name, module in layers.items()]
    
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        
        eigenvalues = None
        vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
            tmp_eigenvalues = [ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors


def get_grouped_layer_weights(model):
    layers = get_layers(model)
    weights = []
    trace_num = []
    layer_names = []
    for name, module in layers.items():
        layer_names.append(name)
        weights.append(module.weight)

    grouped_layer_weights = []
    grouped_layer_names = []
    # Embedding layer
    #grouped_layer_weights.append([weights[0], weights[1]])
    #grouped_layer_names.append('Embedding')
    # Transformer layers
    #grouped_layer_weights.append(weights[:-1])
    #grouped_layer_names.append('Entire model')
    # Entire model
    grouped_layer_weights.append(weights)
    grouped_layer_names.append('Entire model')
    # Head
    grouped_layer_weights.append([weights[-1]])
    grouped_layer_names.append('Head')

    return weights, layer_names, grouped_layer_weights, grouped_layer_names

def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(p.requires_grad for p in module.parameters(recurse=False)):
            #if (type(module) == torch.nn.Linear) and "LayerNorm" not in name and "ln" not in name and "embeddings" not in name and "pooler" not in name:
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
            #print(f"Layer: {name}, Module: {module}")
                layers[name] = module
    return layers

def normalization_list(vs):
    total_norm_sq = sum((v**2).sum() for v in vs)
    norm = total_norm_sq.sqrt()
    if norm < 1e-12:
        return [v.clone() for v in vs]
    return [v / norm for v in vs]

def orthnormal_list(vs, eigenvectors):
    if len(eigenvectors) == 0:
        return normalization_list(vs)
    for e in eigenvectors:
        # e is a list of tensors; compute dot(vs, e)
        dot = sum(torch.sum(v * e_part) for v, e_part in zip(vs, e))
        vs = [v - dot * e_part for v, e_part in zip(vs, e)]
    return normalization_list(vs)

def orthnormal(ws, vs_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for vs in vs_list:
        for w, v in zip(ws, vs):
            w.data.add_(-v*(torch.sum(w*v)))
    return normalization(ws)

# copy from pyhessian
def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def normalization_(vs, epsilon=1e-6):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    """
    norms = [torch.sum(v*v) for v in vs]
    norms = [(norm**0.5).cpu().item() for norm in norms]
    vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
    return vs
    """
    return [v / (torch.norm(v) + epsilon) for v in vs]

def lanczos(A, n_iter):
    n = A.shape[0]
    v = torch.randn(n)

    v_list = []
    alpha_list = []
    beta_list = []

    v = v / torch.linalg.norm(v)
    v_list.append(v)

    # Hessian product vector
    w = A @ v
    alpha_list.append(torch.dot(v, w).item())
    w = w - alpha_list[0] * v

    for j in range(1, n_iter):
        beta_tmp = torch.linalg.norm(w)
        if beta_tmp < 1e-10:
            print("break!")
            # or?
            return alpha_list, beta_list
        beta_list.append(beta_tmp.item())
        v = w / beta_list[-1]
        v_list.append(v)
        w = A @ v
        alpha_list.append(torch.dot(w, v).item())
        w = w - alpha_list[-1] * v - beta_list[-1] * v_list[-2]
    
    return alpha_list, beta_list

def flatten_tensors(tensor_list):
    flats = []
    for t in tensor_list:
        flats.append(t.contiguous().view(-1))
    return torch.cat(flats)

def unflatten_tensors(flat_tensor, tensor_list):
    new_tensors = []
    offset = 0
    for t in tensor_list:
        numel = t.numel()
        new_tensors.append(flat_tensor[offset : offset + numel].view_as(t))
        offset += numel
    return new_tensors

def lanczos_gradient_single(model, loss, weights, n_iter):
    n = sum([w.numel() for w in weights])
    v = torch.randn(n)
    #v = torch.randn_like(weight[0])
    # This gives you a single tensor holding all the gradients
    v_list = []
    alpha_list = []
    beta_list = []

    v = v / torch.linalg.norm(v)
    v_list.append(v)

    # Hessian product vector
    #w = torch.autograd.grad(gradient, weight, grad_outputs=v_list, only_inputs=True, retain_graph=True)
    w = hessian_vector_product_with_tensor_input(model, loss, weights, v)

    alpha_list.append(torch.dot(v, w).item())
    w = w - alpha_list[0] * v

    for j in range(1, n_iter):
        beta_tmp = torch.linalg.norm(w)
        if beta_tmp < 1e-10:
            print("break!")
            # or?
            return alpha_list, beta_list
        beta_list.append(beta_tmp.item())
        v = w / beta_list[-1]
        v_list.append(v)
        #w = torch.autograd.grad(grads, weight, grad_outputs=v, only_inputs=True, retain_graph=True)
        model.zero_grad()
        w = hessian_vector_product_with_tensor_input(model, loss, weights, v)
        alpha_list.append(torch.dot(w, v).item())
        w = w - alpha_list[-1] * v - beta_list[-1] * v_list[-2]
    
    return alpha_list, beta_list

def hessian_vector_product_with_tensor_input(model, loss, weights, d_tensor):
    'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'

    d_tensor = torch.tensor(d_tensor).cuda()
    total_hd_tensor = 0

    t_hd = time.time()

    loss.backward(create_graph= True)
    g_list = []

    for w in weights:
        if w.requires_grad:
            #print(w)
            g_list.append(torch.flatten(w.grad))

    g_tensor = torch.cat(g_list, dim = 0)
    
    model.zero_grad(set_to_none = True)
    g_tensor = g_tensor.cuda()
    l = torch.sum(g_tensor*d_tensor)
    l.backward(retain_graph = True)

    hd_list = []
    for w in weights:
        if w.requires_grad:
            #print(w)
            hd_list.append(torch.flatten(w.grad.data.clone()))

    hd_tensor = torch.cat(hd_list, dim = 0)
    model.zero_grad(set_to_none = True)
    hd_tensor = hd_tensor.cpu()
    total_hd_tensor += hd_tensor
    #print("===;", total_hd_tensor.shape)

    return total_hd_tensor

def weighted_quantile(values, weights, quantile):
    """
    Compute the weighted quantile of a tensor.
    Args:
      values: 1D tensor of eigenvalues.
      weights: 1D tensor of corresponding weights.
      quantile: desired quantile (between 0 and 1).
    Returns:
      The eigenvalue threshold corresponding to the weighted quantile.
    """
    # Sort values and weights in ascending order.
    sorted_vals, sorted_indices = torch.sort(values)
    sorted_weights = weights[sorted_indices]
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = sorted_weights.sum()
    normalized_cum_weights = cumulative_weights / total_weight
    # Find the first index where cumulative weight exceeds the quantile.
    idx = torch.nonzero(normalized_cum_weights >= quantile, as_tuple=False)[0]
    threshold = sorted_vals[idx]
    return threshold

def tail_mass_fraction(values, weights, quantile=0.9):
    """
    Compute the tail mass fraction: the fraction of the total weighted mass
    (∑ p_i λ_i) that comes from eigenvalues above the weighted quantile threshold.
    """
    weights = weights / weights.sum()
    tau = weighted_quantile(values, weights, quantile)
    mask = values >= tau
    numerator = torch.sum(weights[mask] * values[mask])
    denominator = torch.sum(weights * values)
    return (numerator / denominator).item()

def weighted_gini(values, weights):
    """
    Compute the weighted Gini coefficient for the eigenvalue distribution.
    First, normalize weights to obtain a probability distribution:
      q_i = weights_i / (∑_j weights_j).
    Then, compute:
      G = (∑_{i,j} q_i q_j |λ_i - λ_j|) / (2 μ),
    where μ = ∑_i q_i λ_i.
    """
    # Normalize the weights to form a probability distribution.
    q = weights / weights.sum()
    mu = (values * q).sum()
    # Compute pairwise absolute differences between eigenvalues.
    diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
    # Compute pairwise product of normalized weights.
    q_matrix = q.unsqueeze(0) * q.unsqueeze(1)
    gini = torch.sum(diff_matrix * q_matrix) / (2 * mu)
    return gini.item()

def weighted_skewness(values, weights, eps=1e-8):
    """
    Compute the weighted skewness for the eigenvalue distribution.
    Using normalized weights q_i = weights_i / (∑_j weights_j), we have:
      μ   = ∑_i q_i λ_i,
      σ²  = ∑_i q_i (λ_i - μ)²,
      skew = ∑_i q_i (λ_i - μ)³ / (σ³ + eps).
    """
    q = weights / weights.sum()
    mu = (values * q).sum()
    diff = values - mu
    variance = (q * diff**2).sum()
    std = torch.sqrt(variance + eps)
    skew = (q * diff**3).sum() / (std**3 + eps)
    return skew.item()

def compute_sigma_from_weights(state_dict, factor=1.0):
    """
    Compute sigma as a factor times the average standard deviation of the floating-point parameters.
    """
    sigmas = []
    for key, param in state_dict.items():
        if param.requires_grad:
            sigmas.append(param.std().item())
    if sigmas:
        return factor * (sum(sigmas) / len(sigmas))
    else:
        return factor

def compute_kl_divergence_initial_state(final_state_dict, init_state_dict):
    """
    Compute KL(Q||P) where
        Q = N(w_T, sigma^2 I) is the posterior (final weights),
        P = N(w_0, sigma0^2 I) is the prior (initial weights).
    """
    sigma = compute_sigma_from_weights(final_state_dict, factor=0.5)
    sigma0 = compute_sigma_from_weights(init_state_dict, factor=1.0)

    sigma2 = sigma ** 2
    sigma0_2 = sigma0 ** 2
    kl_total = 0.0

    # Loop over parameters (assuming both state_dicts have the same keys)
    for key in final_state_dict:
        param_final = final_state_dict[key]
        param_init = init_state_dict[key].to(param_final.device)
        
        # Consider only floating point parameters (learnable weights)
        if not torch.is_floating_point(param_final) or not torch.is_floating_point(param_init):
            continue
        
        d = param_final.numel()  # number of elements in this tensor
        # Compute squared difference between final and initial weights
        diff_norm_sq = torch.sum((param_final - param_init) ** 2)
        
        # KL divergence for this tensor:
        # KL = 0.5 * [d*log(sigma0^2/sigma^2) + ||w_T - w_0||^2/sigma0^2 + d*(sigma^2/sigma0^2) - d]
        kl_tensor = 0.5 * (d * math.log(sigma0_2 / sigma2) +
                           diff_norm_sq / sigma0_2 +
                           d * (sigma2 / sigma0_2) - d)
        kl_total += kl_tensor

    return kl_total

def pac_bayes_term(kl_div, n, delta):
    """
    Computes the PAC-Bayes bound of the form:
        E[L(f)] <= E[hat{L}(f)] + sqrt((KL(Q||P) + log(1/delta_prime)) / (2n))

    Args:
        empirical_loss (float or torch.Tensor): Empirical loss (averaged over n samples).
        kl_div (float or torch.Tensor): KL(Q||P) already computed.
        n (int): Number of samples in the dataset.
        delta_prime (float): Confidence parameter (e.g., 0.05).

    Returns:
        torch.Tensor: The PAC-Bayes upper bound on the true (expected) loss.
    """
    # Ensure all inputs are torch.Tensor for consistency
    if not isinstance(kl_div, torch.Tensor):
        kl_div = torch.tensor(float(kl_div), dtype=torch.float32)
        
    # Convert n and delta_prime to Tensors if needed
    n_t = torch.tensor(float(n), dtype=torch.float32)
    delta_t = torch.tensor(float(delta), dtype=torch.float32)
    
    # Compute the PAC-Bayes complexity term
    # sqrt( (KL + log(1/delta)) / (2n) )
    complexity_term = torch.sqrt((kl_div + torch.log(1.0 / delta_t)) / (2.0 * n_t))

    # Final bound is empirical_loss + complexity
    return complexity_term

def plot_curves(log, data_names, path_name, file_name=None, save_dir="./results/", x_log=True, y_log=True):
    if file_name is None:
        file_name = path_name
    train_converge = log["train_converge"]["value"]
    val_converge = log["val_converge"]["value"]
    grok_start = log["grok_start"]['value']
    #print(train_converge, val_converge)
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Plotting hessian, ", log.label)
    for i, name in enumerate(data_names):
        plt.plot(log[name]["iter"], log[name]["value"], label=name)

    if train_converge > 0:
        plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
    if val_converge > 0:
        plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Hessian")
    if x_log:
        plt.xscale("log", base=10)
    if y_log:
        plt.yscale("log", base=10)
    #plt.ylim(1e-7, 1e7)
    plt.grid()
    plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
    plt.savefig(f"{save_dir}{path_name}/{file_name}_{log.label}.png", dpi=150)
    plt.draw()
    plt.close()