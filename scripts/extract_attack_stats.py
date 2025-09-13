#!/usr/bin/env python3
"""
Extract attack statistics from YAML files and create a simple CSV for plotting.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

def extract_numpy_values(obj):
    """Extract numpy values from YAML objects."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [extract_numpy_values(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: extract_numpy_values(v) for k, v in obj.items()}
    else:
        return obj

def main():
    """Extract attack statistics and create CSV."""
    stats_data = []
    
    # Load baseline attack stats
    baseline_file = Path("data/adversarial/genetic_attack/attack_statistics.yaml")
    if baseline_file.exists():
        try:
            with open(baseline_file, 'r') as f:
                stats = yaml.load(f, Loader=yaml.FullLoader)
                if stats:
                    # Extract numpy values
                    clean_stats = extract_numpy_values(stats)
                    clean_stats['iteration'] = 0
                    stats_data.append(clean_stats)
                    print(f"Loaded baseline stats: {clean_stats}")
        except Exception as e:
            print(f"Error loading baseline stats: {e}")
    
    # Load iterative training stats
    iter_dir = Path("data/adversarial/iterative_training")
    if iter_dir.exists():
        for i in range(1, 10):
            stats_file = iter_dir / f"attack_stats_iter_{i}.yaml"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = yaml.load(f, Loader=yaml.FullLoader)
                        if stats:
                            clean_stats = extract_numpy_values(stats)
                            clean_stats['iteration'] = i
                            stats_data.append(clean_stats)
                            print(f"Loaded iteration {i} stats: {clean_stats}")
                except Exception as e:
                    print(f"Error loading iteration {i} stats: {e}")
    
    if stats_data:
        df = pd.DataFrame(stats_data)
        output_file = "results/adversarial_training/attack_statistics.csv"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved attack statistics to: {output_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("No attack statistics found")

if __name__ == "__main__":
    main()
