#!/usr/bin/env python3
"""
Utility functions for computing and managing sensitive/insensitive indices across different training phases.
"""

import json
import os
from typing import Dict, List, Set, Tuple, Union
import torch
from taskHessian.datamodels import filter_insensitive_samples


def compute_and_save_phase_indices(
    subset_scores: torch.Tensor,
    phase: int,
    model_name: str,
    result_path: str,
    variance_threshold: float = 1e-2,
    datamodels_num: int = 0,
    save_type: str = "both"
) -> Dict[str, torch.Tensor]:
    """
    Compute and save sensitive/insensitive indices for a specific phase.
    
    Args:
        subset_scores: Shape (M, N) tensor where M is number of training subsets, N is number of test samples
        phase: Phase number (1, 2, or 3)
        model_name: Name of the model (e.g., 'model', 'nso_model')
        result_path: Directory to save the indices files
        variance_threshold: Threshold for variance-based filtering
        datamodels_num: Number of datamodels to use for filtering
        save_type: What to save - "sensitive", "insensitive", or "both"
    
    Returns:
        Dictionary containing the filter results
    """
    # Compute sensitive/insensitive indices for current phase
    filter_results = filter_insensitive_samples(
        subset_scores, 
        variance_threshold=variance_threshold, 
        datamodels_num=datamodels_num
    )
    
    # Save indices based on save_type
    if save_type in ["sensitive", "both"]:
        sensitive_indices = filter_results['sensitive_indices']
        sensitive_file = os.path.join(result_path, f'{model_name}_phase_{phase}_sensitive_indices.json')
        with open(sensitive_file, 'w') as f:
            json.dump(sensitive_indices.tolist(), f)
        print(f"Saved sensitive indices for phase {phase}: {len(sensitive_indices)} indices")
    
    if save_type in ["insensitive", "both"]:
        insensitive_indices = filter_results['insensitive_indices']
        insensitive_file = os.path.join(result_path, f'{model_name}_phase_{phase}_insensitive_indices.json')
        with open(insensitive_file, 'w') as f:
            json.dump(insensitive_indices.tolist(), f)
        print(f"Saved insensitive indices for phase {phase}: {len(insensitive_indices)} indices")
    
    return filter_results


def load_phase_indices(
    phases: List[int],
    model_name: str,
    result_path: str,
    index_type: str = "sensitive"
) -> Tuple[List[Set[int]], List[int]]:
    """
    Load indices from multiple phases.
    
    Args:
        phases: List of phase numbers to load
        model_name: Name of the model
        result_path: Directory containing the indices files
        index_type: Type of indices to load - "sensitive" or "insensitive"
    
    Returns:
        Tuple of (list of index sets, list of available phases)
    """
    all_phases_indices = []
    available_phases = []
    
    for phase in phases:
        phase_file = os.path.join(result_path, f'{model_name}_phase_{phase}_{index_type}_indices.json')
        if os.path.exists(phase_file):
            with open(phase_file, 'r') as f:
                phase_indices = json.load(f)
                all_phases_indices.append(set(phase_indices))
                available_phases.append(phase)
                print(f"Loaded phase {phase} {index_type} indices: {len(phase_indices)} indices")
        else:
            print(f"Warning: {index_type} indices file for phase {phase} not found: {phase_file}")
    
    return all_phases_indices, available_phases


def compute_indices_intersection(
    all_phases_indices: List[Set[int]],
    available_phases: List[int],
    model_name: str,
    result_path: str,
    index_type: str = "sensitive"
) -> List[int]:
    """
    Compute intersection of indices across multiple phases.
    
    Args:
        all_phases_indices: List of sets containing indices from each phase
        available_phases: List of available phase numbers
        model_name: Name of the model
        result_path: Directory to save the intersected indices
        index_type: Type of indices - "sensitive" or "insensitive"
    
    Returns:
        List of intersected indices (sorted)
    """
    if len(all_phases_indices) > 1:
        intersected_indices = set.intersection(*all_phases_indices)
        intersected_indices = sorted(list(intersected_indices))
        print(f"Intersection of {index_type} indices across phases {available_phases}: {len(intersected_indices)} indices")
        
        # Save intersected indices
        intersected_file = os.path.join(result_path, f'{model_name}_intersected_{index_type}_indices.json')
        with open(intersected_file, 'w') as f:
            json.dump(intersected_indices, f)
        print(f"Saved intersected indices to: {intersected_file}")
        
        return intersected_indices
    else:
        print(f"Only phase {available_phases[0]} available, using its indices")
        return sorted(list(all_phases_indices[0]))


def get_intersected_indices_across_phases(
    subset_scores: torch.Tensor,
    current_phase: int,
    model_name: str,
    result_path: str,
    phases_to_check: List[int] = [1, 2, 3],
    variance_threshold: float = 1e-2,
    datamodels_num: int = 0,
    index_type: str = "sensitive"
) -> Tuple[List[int], torch.Tensor]:
    """
    Complete workflow to compute current phase indices, load all available phases,
    and return intersected indices along with filtered subset scores.
    
    Args:
        subset_scores: Shape (M, N) tensor where M is number of training subsets, N is number of test samples
        current_phase: Current phase number
        model_name: Name of the model
        result_path: Directory for saving/loading indices files
        phases_to_check: List of phases to check for intersection
        variance_threshold: Threshold for variance-based filtering
        datamodels_num: Number of datamodels to use for filtering
        index_type: Type of indices to work with - "sensitive" or "insensitive"
    
    Returns:
        Tuple of (intersected indices list, filtered subset scores tensor)
    """
    # Compute and save indices for current phase
    os.makedirs(result_path, exist_ok=True)
    compute_and_save_phase_indices(
        subset_scores=subset_scores,
        phase=current_phase,
        model_name=model_name,
        result_path=result_path,
        variance_threshold=variance_threshold,
        datamodels_num=datamodels_num,
        save_type=index_type
    )
    
    # Load indices from all available phases
    all_phases_indices, available_phases = load_phase_indices(
        phases=phases_to_check,
        model_name=model_name,
        result_path=result_path,
        index_type=index_type
    )
    
    # Compute intersection
    intersected_indices = compute_indices_intersection(
        all_phases_indices=all_phases_indices,
        available_phases=available_phases,
        model_name=model_name,
        result_path=result_path,
        index_type=index_type
    )
    
    # Filter subset scores using the intersected indices
    filtered_subset_scores = subset_scores[:, intersected_indices]
    print(f"Subset scores shape after filtering: {filtered_subset_scores.shape}")
    print(f"Using {len(intersected_indices)} {index_type} indices for datamodels computation")
    
    return intersected_indices, filtered_subset_scores


def load_intersected_indices(
    model_name: str,
    result_path: str,
    index_type: str = "sensitive"
) -> Union[List[int], None]:
    """
    Load previously computed intersected indices.
    
    Args:
        model_name: Name of the model
        result_path: Directory containing the indices files
        index_type: Type of indices - "sensitive" or "insensitive"
    
    Returns:
        List of intersected indices or None if file doesn't exist
    """
    intersected_file = os.path.join(result_path, f'{model_name}_intersected_{index_type}_indices.json')
    if os.path.exists(intersected_file):
        with open(intersected_file, 'r') as f:
            indices = json.load(f)
        print(f"Loaded intersected {index_type} indices: {len(indices)} indices")
        return indices
    else:
        print(f"Intersected {index_type} indices file not found: {intersected_file}")
        return None


def get_phase_indices_summary(
    phases: List[int],
    model_name: str,
    result_path: str,
    index_type: str = "sensitive"
) -> Dict[str, any]:
    """
    Get a summary of indices across different phases.
    
    Args:
        phases: List of phase numbers to analyze
        model_name: Name of the model
        result_path: Directory containing the indices files
        index_type: Type of indices - "sensitive" or "insensitive"
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'available_phases': [],
        'phase_counts': {},
        'intersection_count': 0,
        'union_count': 0,
        'total_unique_indices': 0
    }
    
    all_phases_indices, available_phases = load_phase_indices(
        phases=phases,
        model_name=model_name,
        result_path=result_path,
        index_type=index_type
    )
    
    summary['available_phases'] = available_phases
    
    if all_phases_indices:
        # Phase-wise counts
        for i, phase in enumerate(available_phases):
            summary['phase_counts'][f'phase_{phase}'] = len(all_phases_indices[i])
        
        # Intersection count
        if len(all_phases_indices) > 1:
            intersection = set.intersection(*all_phases_indices)
            summary['intersection_count'] = len(intersection)
        
        # Union count
        union = set.union(*all_phases_indices)
        summary['union_count'] = len(union)
        summary['total_unique_indices'] = len(union)
    
    return summary
