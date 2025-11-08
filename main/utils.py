"""
Utility functions for managing results and directories.
"""

import os
from pathlib import Path


def get_next_run_dir(base_dir='results/runs', prefix='run'):
    """
    Get next available run directory with auto-incremented number.
    
    Args:
        base_dir: Base directory for runs
        prefix: Prefix for run directories
    
    Returns:
        Path to next run directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Find existing run directories
    existing_runs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    
    if not existing_runs:
        next_num = 1
    else:
        # Extract numbers from existing runs
        numbers = []
        for run_dir in existing_runs:
            try:
                num = int(run_dir.name.replace(prefix + '_', ''))
                numbers.append(num)
            except ValueError:
                continue
        
        next_num = max(numbers) + 1 if numbers else 1
    
    # Create next run directory
    next_run = base_path / f"{prefix}_{next_num:03d}"
    next_run.mkdir(parents=True, exist_ok=True)
    
    return next_run


def ensure_results_structure():
    base_path = Path('results')
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / 'models').mkdir(exist_ok=True)
    (base_path / 'runs').mkdir(exist_ok=True)
    (base_path / 'attacks').mkdir(exist_ok=True)


