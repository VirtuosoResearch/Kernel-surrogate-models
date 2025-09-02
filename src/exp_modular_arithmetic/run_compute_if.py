#!/usr/bin/env python3
"""
Script to run compute_if.py with different configurations
This demonstrates how to compute influence functions for trained models
"""

import subprocess
import sys
import os

def run_compute_if(config):
    """Run compute_if.py with the given configuration"""
    cmd = [
        sys.executable, "src/exp_modular_arithmetic/compute_if.py",
        "--task", config["task"],
        "--p", str(config["p"]),
        "--train_ratio", str(config["train_ratio"]),
        "--valid_ratio", str(config["valid_ratio"]),
        "--device", str(config["device"]),
        "--model_type", config["model_type"],
        "--mark", config["mark"]
    ]
    
    # Add optimizer flags
    if config.get("sam"):
        cmd.append("--sam")
    if config.get("nsm"):
        cmd.append("--nsm")
    if config.get("reg"):
        cmd.append("--reg")
    if config.get("swa"):
        cmd.append("--swa")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Main function to run different configurations"""
    
    # Change to the project root directory
    project_root = "/data/zhenshuo/task-Modeling-and-Influence-Function"
    os.chdir(project_root)
    
    print("Running compute_if.py to compute influence functions...")
    print("Make sure you have trained models first using train_all.py")
    print("=" * 60)
    
    # Configuration 1: Standard training (no special optimizer)
    config1 = {
        "task": "addition",
        "p": 97,
        "train_ratio": 0.9,
        "valid_ratio": 0.1,
        "device": 0,
        "model_type": "tf",
        "mark": ""
    }
    
    print("\n1. Standard training model")
    success1 = run_compute_if(config1)
    
    # Configuration 2: SAM optimizer
    config2 = {
        "task": "addition",
        "p": 97,
        "train_ratio": 0.9,
        "valid_ratio": 0.1,
        "device": 0,
        "model_type": "tf",
        "sam": True,
        "mark": ""
    }
    
    print("\n2. SAM optimizer model")
    success2 = run_compute_if(config2)
    
    # Configuration 3: NSM optimizer
    config3 = {
        "task": "addition",
        "p": 97,
        "train_ratio": 0.9,
        "valid_ratio": 0.1,
        "device": 0,
        "model_type": "tf",
        "nsm": True,
        "mark": ""
    }
    
    print("\n3. NSM optimizer model")
    success3 = run_compute_if(config3)
    
    # Configuration 4: MLP model
    config4 = {
        "task": "addition",
        "p": 97,
        "train_ratio": 0.9,
        "valid_ratio": 0.1,
        "device": 0,
        "model_type": "mlp",
        "mark": ""
    }
    
    print("\n4. MLP model")
    success4 = run_compute_if(config4)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Standard training: {'✓' if success1 else '✗'}")
    print(f"SAM optimizer: {'✓' if success2 else '✗'}")
    print(f"NSM optimizer: {'✓' if success3 else '✗'}")
    print(f"MLP model: {'✓' if success4 else '✗'}")
    
    if all([success1, success2, success3, success4]):
        print("\n🎉 All influence function computations completed successfully!")
    else:
        print("\n⚠️  Some computations failed. Check the output above for details.")
    
    print("Results are saved in ./results/influence_function/")

if __name__ == "__main__":
    main()
