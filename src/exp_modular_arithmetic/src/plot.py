import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def draw_curve(results):
    args = results['args']
    its = results['its']
    steps = results['steps']
    val_steps = results['val_steps']
    train_acc = results['train_acc']
    train_loss = results['train_loss']
    val_acc = results['val_acc']
    val_loss = results['val_loss']
    loss_gap = results['loss_gap']
    train_converge = results['train_converge']
    val_converge = results['val_converge']
    hessian_its = results['hessian_its']
    # train_hessiantrace = results['train_hessiantrace']
    # val_hessiantrace = results['val_hessiantrace']
    train_hessianmeasurement = results['train_hessianmeasurement']
    # train_hessiandistance = results['train_hessiandistance']
    # val_hessianmeasurement = results['train_hessianmeasurement']
    # train_largestHeigen = results['train_largestHeigen']
    # train_largestratio = results['train_largestratio']
    # loss_gap_cor_s1 = results['loss_gap_cor_s1']
    # loss_gap_cor_s2 = results['loss_gap_cor_s2']
    # loss_gap_cor_s3 = results['loss_gap_cor_s3']
    # hessian_measure_cor_s1 = results['hessian_measure_cor_s1']
    # hessian_measure_cor_s2 = results['hessian_measure_cor_s2']
    # hessian_measure_cor_s3 = results['hessian_measure_cor_s3']

    # Accuracy
    plt.plot(steps, train_acc, label="train")
    plt.plot(val_steps, val_acc, label="val")
    if train_converge > 0:
        plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
    if val_converge > 0:
        plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
    plt.legend()
    plt.title(f"{args.task}")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Accuracy")
    plt.xscale("log", base=10)
    plt.grid()
    acc_str = f"best train acc: {np.array(train_acc).max():.4f}\nvalid acc: {val_acc[-1]:.4f}\nbest valid acc: {np.array(val_acc).max():.4f}\ngrokking gap: {val_converge-train_converge}"
    plt.annotate(acc_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
    plt.savefig(f"./results/acc/acc_{args.label}.png", dpi=150)
    plt.show()
    plt.close()

    # Loss
    plt.plot(steps, train_loss, label="train")
    plt.plot(val_steps, val_loss, label="val")
    if train_converge > 0:
        plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
    if val_converge > 0:
        plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
    plt.legend()
    plt.title(f"{args.task}")
    plt.xlabel("Optimization Steps", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.xscale("log", base=10)
    plt.grid()
    plt.savefig(f"./results/loss/loss_{args.label}.png", dpi=150)
    plt.show()
    plt.close()

    if args.hessian_save_every > 0:
        """
        # Hessian trace
        plt.plot(hessian_its, [abs(trace) for trace in train_hessiantrace], label="train")
        plt.plot(hessian_its, [abs(trace) for trace in val_hessiantrace], label="val")
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        #if args.sam == True:
        plt.plot(hessian_its, train_largestHeigen, label="lambda1")
        plt.legend()
        plt.title(f"{args.task}")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Hessian trace")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.ylim(1e-3, 1e8)
        plt.grid()
        plt.savefig(f"./results/hessian/hessiantrace_{args.label}.png", dpi=150)
        plt.show()
        plt.close()   
        """
        # Gap
        plt.plot(hessian_its[1:], loss_gap, label="loss gap")
        plt.plot(hessian_its, train_hessianmeasurement, label="train hessian measurement")
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        plt.legend()
        plt.title(f"{args.task}")
        plt.xlabel("Gap")
        plt.ylabel("Hessian")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        #plt.ylim(1e-7, 1e7)
        plt.grid()
        plt.savefig(f"./results/gap/gap_{args.label}.png", dpi=150)
        plt.draw()
        plt.close()
        """
        # Correlation
        plt.scatter(hessian_measure_cor_s1, loss_gap_cor_s1, color='red', label="Stage 1")
        plt.scatter(hessian_measure_cor_s2, loss_gap_cor_s2, color='green', label="Stage 2")
        plt.scatter(hessian_measure_cor_s3, loss_gap_cor_s3, color='blue', label="Stage 3")
        plt.legend()
        plt.title(f"{args.task}")
        plt.xlabel("Hessian measurement")
        plt.ylabel("Loss gap")
        #plt.xscale("log", base=10)
        #plt.yscale("log", base=10)
        #plt.ylim(1e-3, 1e8)
        plt.grid()
        plt.savefig(f"./results/cor/correlation_{args.label}.png", dpi=150)
        plt.close()
        """
        """
        # ratio
        if train_converge > 0:
            plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
        if val_converge > 0:
            plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
        #if args.sam == True:
        plt.plot(hessian_its, train_largestratio, label="ratio")
        plt.legend()
        plt.title(f"{args.task}")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Ratio of largest eigenvalue")
        plt.xscale("log", base=10)
        #plt.yscale("log", base=10)
        
        plt.grid()
        plt.savefig(f"./results/ratio/ratio_{args.label}.png", dpi=150)
        plt.show()
        plt.close()
        """
        