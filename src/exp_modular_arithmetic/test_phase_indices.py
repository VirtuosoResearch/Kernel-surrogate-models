#!/usr/bin/env python3
"""
Test script to demonstrate usage of phase_indices_utils.py
"""

import torch
import numpy as np
import os
import tempfile
from phase_indices_utils import (
    compute_and_save_phase_indices,
    load_phase_indices,
    compute_indices_intersection,
    get_intersected_indices_across_phases,
    get_phase_indices_summary
)

def create_test_data(num_subsets=50, num_test_samples=1000):
    """Create synthetic test data."""
    # Create subset scores with some variance patterns
    subset_scores = torch.randn(num_subsets, num_test_samples)
    
    # Make some columns have very low variance (insensitive)
    low_variance_cols = np.random.choice(num_test_samples, size=200, replace=False)
    for col in low_variance_cols:
        subset_scores[:, col] = torch.normal(0.5, 0.001, (num_subsets,))
    
    return subset_scores

def test_phase_indices_utils():
    """Test the phase indices utility functions."""
    print("Testing phase indices utility functions...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test data
        subset_scores = create_test_data()
        print(f"Created test data with shape: {subset_scores.shape}")
        
        # Test computing and saving indices for different phases
        for phase in [1, 2, 3]:
            print(f"\n--- Testing Phase {phase} ---")
            
            # Add some noise to make phases slightly different
            if phase > 1:
                noise = torch.randn_like(subset_scores) * 0.01
                test_scores = subset_scores + noise
            else:
                test_scores = subset_scores
            
            # Compute and save indices
            filter_results = compute_and_save_phase_indices(
                subset_scores=test_scores,
                phase=phase,
                model_name="test_model",
                result_path=temp_dir,
                variance_threshold=1e-2,
                datamodels_num=10,
                save_type="both"
            )
            
            print(f"Phase {phase} - Sensitive: {len(filter_results['sensitive_indices'])}, "
                  f"Insensitive: {len(filter_results['insensitive_indices'])}")
        
        # Test loading indices from multiple phases
        print("\n--- Testing Index Loading ---")
        all_phases_indices, available_phases = load_phase_indices(
            phases=[1, 2, 3],
            model_name="test_model",
            result_path=temp_dir,
            index_type="sensitive"
        )
        
        # Test intersection computation
        print("\n--- Testing Intersection ---")
        intersected_indices = compute_indices_intersection(
            all_phases_indices=all_phases_indices,
            available_phases=available_phases,
            model_name="test_model",
            result_path=temp_dir,
            index_type="sensitive"
        )
        
        # Test the complete workflow
        print("\n--- Testing Complete Workflow ---")
        final_indices, filtered_scores = get_intersected_indices_across_phases(
            subset_scores=subset_scores,
            current_phase=1,
            model_name="workflow_test",
            result_path=temp_dir,
            phases_to_check=[1, 2, 3],
            variance_threshold=1e-2,
            datamodels_num=10,
            index_type="insensitive"
        )
        
        print(f"Final filtered scores shape: {filtered_scores.shape}")
        print(f"Number of intersected indices: {len(final_indices)}")
        
        # Test summary function
        print("\n--- Testing Summary ---")
        summary = get_phase_indices_summary(
            phases=[1, 2, 3],
            model_name="test_model",
            result_path=temp_dir,
            index_type="sensitive"
        )
        
        print("Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_phase_indices_utils()
