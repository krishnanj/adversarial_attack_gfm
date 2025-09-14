#!/usr/bin/env python3
"""
Script to run iterative adversarial training pipeline.

This script implements the complete iterative adversarial training process:
1. Train initial model on original data
2. Generate adversarial examples using genetic algorithm
3. Retrain model on original + adversarial data
4. Repeat for specified iterations
5. Monitor and plot robustness improvements

Usage:
    python scripts/run_adversarial_training.py [--dataset promoter|tf] [--test]
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adversarial_training import IterativeAdversarialTrainer
from utils import setup_logging
import logging

def main():
    """Main function to run iterative adversarial training."""
    parser = argparse.ArgumentParser(description='Run iterative adversarial training')
    parser.add_argument('--dataset', type=str, choices=['promoter', 'tf'], 
                       default='promoter', help='Dataset to use (default: promoter)')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode with reduced parameters')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration file
    config_path = "configs/adversarial_training.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    # Load and modify config based on dataset and test mode
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update dataset
    config['training']['dataset'] = args.dataset
    if args.dataset == 'tf':
        config['training']['max_length'] = 200  # TF sequences are shorter
    
    # Update attack config
    attack_config_path = "configs/attack_genetic.yaml"
    with open(attack_config_path, 'r') as f:
        attack_config = yaml.safe_load(f)
    attack_config['attack']['dataset'] = args.dataset
    
    # Test mode adjustments
    if args.test:
        config['training']['num_epochs'] = 1
        config['training']['warmup_steps'] = 100
        config['adversarial_training']['max_iterations'] = 2
        attack_config['attack']['test_samples'] = 10
        attack_config['genetic_algorithm']['population_size'] = 20
        attack_config['genetic_algorithm']['max_generations'] = 5
        logger.info("🧪 Running in TEST mode with reduced parameters")
    
    # Save modified configs
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(attack_config_path, 'w') as f:
        yaml.dump(attack_config, f, default_flow_style=False)
    
    # Check if test data exists
    if args.dataset == 'promoter':
        test_data_path = "data/raw/GUE/prom/prom_300_all/test.csv"
    else:  # tf
        test_data_path = "data/raw/GUE/tf/0/test.csv"
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found: {test_data_path}")
        logger.error(f"Please ensure the GUE {args.dataset} dataset is available")
        return 1
    
    try:
        logger.info(f"🚀 Starting iterative adversarial training on {args.dataset} dataset...")
        logger.info(f"📋 Using config: {config_path}")
        
        # Initialize trainer
        trainer = IterativeAdversarialTrainer(config_path)
        
        # Run iterative training
        results = trainer.run_iterative_training()
        
        logger.info("✅ Iterative adversarial training completed successfully!")
        logger.info(f"📊 Final results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error in iterative adversarial training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())