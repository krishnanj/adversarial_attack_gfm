#!/usr/bin/env python3
"""
Script to run genetic algorithm-based adversarial attacks on DNABERT-2.

This script loads the trained baseline model and runs genetic algorithm
attacks on the test dataset to evaluate model robustness.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from attack_genetic import GeneticAdversarialAttack


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/genetic_attack.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to run genetic algorithm attack."""
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Genetic Algorithm Adversarial Attack")
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'configs' / 'attack_genetic.yaml'
    test_data_path = base_dir / 'data' / 'raw' / 'GUE' / 'prom' / 'prom_300_all' / 'test.csv'
    
    # Check if files exist
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data file not found: {test_data_path}")
        return
    
    # Check if model exists
    model_path = base_dir / 'models' / 'baseline' / 'checkpoint_epoch_3.pt'
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.error("Please train the baseline model first using: python src/train_forward.py --dataset promoter")
        return
    
    try:
        # Initialize attack
        logger.info("Initializing genetic algorithm attack...")
        attack = GeneticAdversarialAttack(config_path)
        
        # Run attack
        logger.info("Running genetic algorithm attack...")
        results_df, stats = attack.run_attack(test_data_path)
        
        # Print detailed results
        logger.info("\n" + "="*50)
        logger.info("GENETIC ALGORITHM ATTACK RESULTS")
        logger.info("="*50)
        logger.info(f"Total attacks attempted: {stats['total_attacks']}")
        logger.info(f"Successful attacks: {stats['successful_attacks']}")
        logger.info(f"Success rate: {stats['success_rate']:.2%}")
        logger.info(f"Average confidence drop: {stats['avg_confidence_drop']:.3f}")
        logger.info(f"Average perturbations: {stats['avg_perturbations']:.1f}")
        logger.info(f"Average biological score: {stats['avg_biological_score']:.3f}")
        logger.info(f"Average generations used: {stats['avg_generations']:.1f}")
        
        # Show some example results
        logger.info("\nExample successful attacks:")
        successful_attacks = results_df[results_df['attack_successful'] == True]
        if len(successful_attacks) > 0:
            for i, (_, row) in enumerate(successful_attacks.head(3).iterrows()):
                logger.info(f"\nExample {i+1}:")
                logger.info(f"  Original confidence: {row['original_confidence']:.3f}")
                logger.info(f"  Adversarial confidence: {row['adversarial_confidence']:.3f}")
                logger.info(f"  Confidence drop: {row['confidence_drop']:.3f}")
                logger.info(f"  Perturbations: {row['perturbations']}")
                logger.info(f"  Biological score: {row['biological_score']:.3f}")
                logger.info(f"  Generations used: {row['generations_used']}")
        else:
            logger.info("No successful attacks found.")
        
        logger.info(f"\nResults saved to: {attack.config['output']['output_dir']}")
        
    except Exception as e:
        logger.error(f"Error during attack: {e}")
        raise


if __name__ == "__main__":
    main()
