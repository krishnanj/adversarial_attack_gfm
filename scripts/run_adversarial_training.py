#!/usr/bin/env python3
"""
Script to run iterative adversarial training pipeline.

This script implements the complete iterative adversarial training process:
1. Train initial model on original data
2. Generate adversarial examples using genetic algorithm
3. Retrain model on original + adversarial data
4. Repeat for specified iterations
5. Monitor and plot robustness improvements
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from adversarial_training import IterativeAdversarialTrainer
from utils import setup_logging
import logging

def main():
    """Main function to run iterative adversarial training."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration file
    config_path = "configs/adversarial_training.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    # Note: We now start from pretrained model, no need to check for baseline checkpoint
    
    # Check if test data exists
    test_data_path = "data/raw/GUE/prom/prom_300_all/test.csv"
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found: {test_data_path}")
        logger.error("Please ensure the GUE promoter dataset is available")
        return 1
    
    try:
        logger.info("Starting iterative adversarial training pipeline...")
        
        # Initialize trainer
        trainer = IterativeAdversarialTrainer(config_path)
        
        # Run iterative training
        results = trainer.run_iterative_training()
        
        logger.info("Iterative adversarial training completed successfully!")
        logger.info(f"Results saved to: {trainer.config['output']['results_dir']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in iterative adversarial training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
