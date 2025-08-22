import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.stats import norm
from datetime import datetime

from src.configs.config import ConfigDict


def plot_spectral_density(flat_eigen, flat_weight, sigma=0.1, grid_size=100, plot_individual=False, color='blue', file_label='', label='Spectral Density'):
    save_dir = "./results/"
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
            plt.plot(lambdas, gaussian, color='gray', alpha=0.3)
            total_density += gaussian
    else:
        # Sum all contributions without plotting individual ones
        for eig, w in zip(flat_eigen, flat_weight):
            total_density += w * norm.pdf(lambdas, loc=eig, scale=sigma)
    
    # Normalize the total density
    total_density /= np.sum(total_density) * delta_lambda
    
    # Plot the total spectral density
    plt.plot(lambdas, total_density, color=color, linewidth=2, label=label)
    
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
    plt.savefig(f"{save_dir}spectrum/spectrum_{file_label}_{label}.png", dpi=150)
    plt.draw()
    plt.close()

def plot(log, config, save_dir = "./results/"):
    try:
        if config.mark == 'standard':
            save_dir += "standard/"
        # Train accuracy
        train_converge = log["train_converge"]["value"]
        val_converge = log["val_converge"]["value"]
        grok_start = log["grok_start"]['value']
        #print(train_converge, val_converge)
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Accuracy
        train_acc = log['train_acc']['value']
        val_acc = log['val_acc']['value']
        plt.plot(log['train_acc']['iter'], log['train_acc']['value'], label="train")
        plt.plot(log['val_acc']['iter'], log['val_acc']['value'], label="val")
        for key in log.data.keys():
            if "neighbor_acc" in key:
                plt.plot(log[key]['iter'], log[key]['value'], label=key)

        if config.swa and grok_start > 0:
            plt.plot(log['swa_acc']['iter'], log['swa_acc']['value'], label="swa val")
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        if grok_start > 0:
            plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Accuracy")
        plt.xscale("log", base=10)
        plt.grid()
        acc_str = f"best train acc: {np.array(train_acc).max():.4f}\nvalid acc: {val_acc[-1]:.4f}\nbest valid acc: {np.array(val_acc).max():.4f}\ngrokking gap: {val_converge-train_converge}\nval converge: {val_converge}\ntime: {time_str}"
        plt.annotate(acc_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}acc/acc_{log.label}.png", dpi=150)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)
    
    try:
        # Loss
        plt.plot(log['train_loss']['iter'], log['train_loss']['value'], label="train")
        plt.plot(log['val_loss']['iter'], log['val_loss']['value'], label="val")
        for key in log.data.keys():
            if "neighbor_loss" in key:
                plt.plot(log[key]['iter'], log[key]['value'], label=key)

        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        if grok_start > 0:
            plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("Optimization Steps", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.xscale("log", base=10)
        plt.grid()
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}loss/loss_{log.label}.png", dpi=150)
        plt.show()
        plt.close()

        # Log Loss
        plt.plot(log['train_loss']['iter'], log['train_loss']['value'], label="train")
        plt.plot(log['val_loss']['iter'], log['val_loss']['value'], label="val")
        for key in log.data.keys():
            if "neighbor_loss" in key:
                plt.plot(log[key]['iter'], log[key]['value'], label=key)
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        if grok_start > 0:
            plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
        plt.legend()
        plt.title(f"{config.task.task}")
        plt.xlabel("Optimization Steps", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.grid()
        plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
        plt.savefig(f"{save_dir}logloss/logloss_{log.label}.png", dpi=150)
        plt.show()
        plt.close()
    except Exception as e: 
        print(e)
    if config.train.hessian_log_every > 0:
        try:
            # Hessian
            plt.plot(log["train_hessian_trace"]["iter"], log["train_hessian_trace"]["value"], label="Hessian")
            # |Train loss - val loss|
            plt.plot(log["train_wd_hessian_trace"]["iter"], log["train_wd_hessian_trace"]["value"], label="wd Hessian")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Gap")
            plt.ylabel("Hessian")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}hessian/hessian_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # Hessian

            plt.plot(log["train_hessianmeasurement"]["iter"], log["train_hessianmeasurement"]["value"], label="Hessian")
            # |Train loss - val loss|
            plt.plot(log["loss_gap"]["iter"], log["loss_gap"]["value"], label="|Train Loss - Val Loss|")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Gap")
            plt.ylabel("Hessian")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}gap/gap_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # layer hessian trace
            hessian_layer_trace = np.array(log["hessian_layer_trace"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(hessian_layer_trace)):
                plt.plot(log["hessian_layer_trace"]["iter"], hessian_layer_trace[i], label=f"TF-{i} and head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}hessian/hessian_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
            """
            # Hessian lambda
            plt.plot(log["train_hessian_lambda_1"]["iter"], log["train_hessian_lambda_1"]["value"], label="lambda_1")
            plt.plot(log["train_hessian_lambda_2"]["iter"], log["train_hessian_lambda_2"]["value"], label="lambda_2")
            plt.plot(log["train_hessian_condition"]["iter"], log["train_hessian_condition"]["value"], label="1/2")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Epochs")
            plt.ylabel("Lambda")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            lambda_str = f"time: {log['time']['value']}"
            plt.annotate(gap_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}lambda/lambda_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
            """
        except Exception as e: 
            print(e)
        try:
            # cos sim
            plt.plot(log['train_loss_wd_cossim']['iter'], log['train_loss_wd_cossim']['value'], label="train")
            #plt.plot(log['val_loss']['iter'], log['val_loss']['value'], label="val")
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Optimization Steps", fontsize=20)
            plt.ylabel("Cos sim", fontsize=20)
            plt.xscale("log", base=10)
            #plt.yscale("log", base=10)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}loss_wd_sim/loss_wd_sim_{log.label}.png", dpi=150)
            plt.show()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # l2 
            plt.plot(log['train_loss_wd_l2']['iter'], log['train_loss_wd_l2']['value'], label="distance")
            plt.plot(log['train_grad_norm']['iter'], log['train_grad_norm']['value'], label="grad")
            plt.plot(log['train_wd_grad_norm']['iter'], log['train_wd_grad_norm']['value'], label="wd_grad")
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Optimization Steps", fontsize=20)
            plt.ylabel("Cos sim", fontsize=20)
            plt.xscale("log", base=10)
            #plt.yscale("log", base=10)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}loss_wd_dis/loss_wd_dis_{log.label}.png", dpi=150)
            plt.show()
            plt.close()

            """
            # H eigenvector distance
            plt.plot(log['train_hessian_eigendistance']['iter'], log['train_hessian_eigendistance']['value'], label="eigenvector distance")
            #plt.plot(log['val_loss']['iter'], log['val_loss']['value'], label="val")
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Optimization Steps", fontsize=20)
            plt.ylabel("Cos sim", fontsize=20)
            plt.xscale("log", base=10)
            #plt.yscale("log", base=10)
            plt.grid()
            loss_wd_sim_str = f"time: {log['time']['value']}"
            plt.annotate(loss_wd_sim_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}distance/distance_{log.label}.png", dpi=150)
            plt.show()
            plt.close()
            """
        except Exception as e: 
            print(e)
        try:
            # loss singular value distance
            plt.plot(log['train_loss_singularvalue_distance']['iter'], log['train_loss_singularvalue_distance']['value'], label="loss singularvalue ratio")
            #plt.plot(log['val_loss']['iter'], log['val_loss']['value'], label="val")
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Optimization Steps", fontsize=20)
            plt.ylabel("ratio", fontsize=20)
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}distance/distance_{log.label}.png", dpi=150)
            plt.show()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # p(x)-y
            plt.plot(log["train_item_1"]["iter"], log["train_item_1"]["value"], label="p-y")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Gap")
            plt.ylabel("Hessian")
            plt.xscale("log", base=10)
            #plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}prob/prob_1_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # diag(p(x))-p(x)p(x)^T
            plt.plot(log["train_item_2"]["iter"], log["train_item_2"]["value"], label="diag(p)-pp^T")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("Gap")
            plt.ylabel("Hessian")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}prob/prob_2_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        
        try:
            # block sim
            spectrum_divergence = np.array(log["spectrum_divergence"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(spectrum_divergence)):
                plt.plot(log["spectrum_divergence"]["iter"], spectrum_divergence[i], label=f"TF-{i} and head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}block_sim/block_sim_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        """
        try:
            # spectral entropy
            spectral_entropy = np.array(log["spectral_entropy"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(spectral_entropy)-1):
                plt.plot(log["spectral_entropy"]["iter"], spectral_entropy[i], label=f"TF-{i}")
            plt.plot(log["spectral_entropy"]["iter"], spectral_entropy[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/entropy_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # weighted entropy
            weighted_entropy = np.array(log["weighted_entropy"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(weighted_entropy)-1):
                plt.plot(log["weighted_entropy"]["iter"], weighted_entropy[i], label=f"TF-{i}")
            plt.plot(log["weighted_entropy"]["iter"], weighted_entropy[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/weighted_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # centroid entropy
            centroid = np.array(log["centroid"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(centroid)-1):
                plt.plot(log["centroid"]["iter"], centroid[i], label=f"TF-{i}")
            plt.plot(log["centroid"]["iter"], centroid[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/centroid_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        try:
            # centroid entropy
            spread = np.array(log["spread"]["value"]).T
            #print(spectrum_divergence)
            for i in range(len(spread)-1):
                plt.plot(log["spread"]["iter"], spread[i], label=f"TF-{i}")
            plt.plot(log["spread"]["iter"], spread[-1], label=f"head")
            #plt.plot(log["train_block_sim_2"]["iter"], log["train_block_sim_2"]["value"], label="1 and 2")
            #plt.plot(log["train_block_sim_3"]["iter"], log["train_block_sim_3"]["value"], label="0 and 1")
            # Plot
            if train_converge > 0:
                plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
            if val_converge > 0:
                plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
            if grok_start > 0:
                plt.axvline(x=grok_start, color='yellow', linestyle='--', linewidth=1, label='grokking start')
            plt.legend()
            plt.title(f"{config.task.task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            #plt.ylim(1e-7, 1e7)
            plt.grid()
            plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
            plt.savefig(f"{save_dir}entropy/spread_{log.label}.png", dpi=150)
            plt.draw()
            plt.close()
        except Exception as e: 
            print(e)
        """
        
