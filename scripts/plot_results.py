#!/usr/bin/env python3
"""
Script to generate comprehensive plots for iterative adversarial training results.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from plot_results import AdversarialTrainingPlotter

def main():
    """Generate all plots for the paper."""
    print("Generating comprehensive plots for iterative adversarial training...")
    
    plotter = AdversarialTrainingPlotter()
    plotter.generate_all_plots()
    
    print("✅ All plots generated successfully!")
    print(f"📊 Plots saved to: plots/adversarial_training/")
    print(f"📁 Plot data saved to: results/adversarial_training/plot_data/")

if __name__ == "__main__":
    main()
