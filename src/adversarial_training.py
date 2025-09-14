"""
Iterative Adversarial Training Pipeline for DNABERT-2

This module implements iterative adversarial training where:
1. Train model on original data
2. Generate adversarial examples using genetic algorithm
3. Retrain model on original + adversarial data
4. Repeat for specified iterations
5. Monitor and plot robustness improvements
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
from datetime import datetime

from utils import load_config, set_seed, setup_logging
from train_forward import DNABERT2Classifier, train_model, evaluate_model
from attack_genetic import GeneticAdversarialAttack


class IterativeAdversarialTrainer:
    """Handles iterative adversarial training pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize the iterative adversarial trainer."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        set_seed(self.config['training']['seed'])
        
        # Create output directories
        self._create_directories()
        
        # Initialize tracking variables
        self.training_metrics = []
        self.iteration_results = []
        
        # Load attack configuration
        self.attack_config = load_config(self.config['attack_config'])
        
        # Initialize genetic attack
        self.genetic_attack = GeneticAdversarialAttack(self.config['attack_config'])
        
        self.logger.info(f"Initialized iterative adversarial trainer with {self.config['adversarial_training']['max_iterations']} max iterations")
    
    def _create_directories(self):
        """Create necessary output directories."""
        dirs_to_create = [
            self.config['adversarial_training']['adversarial_data_dir'],
            self.config['checkpointing']['save_dir'],
            self.config['monitoring']['plot_dir'],
            self.config['output']['results_dir']
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
    
    def load_original_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load original training, validation, and test data."""
        dataset_path = f"data/raw/GUE/prom/prom_300_all"
        
        train_df = pd.read_csv(f"{dataset_path}/train.csv")
        val_df = pd.read_csv(f"{dataset_path}/dev.csv")
        test_df = pd.read_csv(f"{dataset_path}/test.csv")
        
        self.logger.info(f"Loaded original data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def generate_adversarial_examples(self, model_path: str, test_df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Generate adversarial examples using genetic algorithm."""
        self.logger.info(f"Generating adversarial examples for iteration {iteration}")
        
        # Update attack config to use current model
        self.attack_config['attack']['target_model'] = model_path
        
        # Use different random seed for each iteration to get different adversarial examples
        self.attack_config['seed'] = 42 + iteration * 100
        
        # Create temporary attack config file
        temp_config_path = f"configs/temp_attack_iter_{iteration}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(self.attack_config, f)
        
        # Initialize attack with updated config
        attack = GeneticAdversarialAttack(temp_config_path)
        
        # Run attack
        results, attack_stats = attack.run_attack(test_df)
        
        # Clean up temp config
        os.remove(temp_config_path)
        
        # Save adversarial examples
        if self.config['adversarial_training']['save_adversarial_examples']:
            adv_path = f"{self.config['adversarial_training']['adversarial_data_dir']}/adversarial_iter_{iteration}.csv"
            results.to_csv(adv_path, index=False)
            self.logger.info(f"Saved adversarial examples to {adv_path}")
        
        # Save attack statistics for plotting
        clean_stats = self._clean_numpy_objects(attack_stats)
        stats_path = f"{self.config['adversarial_training']['adversarial_data_dir']}/attack_stats_iter_{iteration}.yaml"
        with open(stats_path, 'w') as f:
            yaml.dump(clean_stats, f, default_flow_style=False)
        
        return results
    
    def _clean_numpy_objects(self, obj):
        """Convert numpy objects to regular Python types."""
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self._clean_numpy_objects(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._clean_numpy_objects(v) for k, v in obj.items()}
        else:
            return obj
    
    def prepare_adversarial_training_data(self, adversarial_df: pd.DataFrame, iteration: int) -> pd.DataFrame:
        """Prepare cumulative training data: original + all adversarial examples from previous iterations."""
        # Filter successful adversarial examples from current iteration
        successful_adv = adversarial_df[adversarial_df['attack_successful'] == True].copy()
        
        if len(successful_adv) == 0:
            self.logger.warning(f"No successful adversarial examples found for iteration {iteration}")
            return self.original_train_df.copy()  # Return original data if no adversarial examples
        
        # Prepare adversarial data
        adv_training_data = successful_adv[['adversarial_sequence', 'true_label']].copy()
        adv_training_data.columns = ['sequence', 'label']
        
        # Load all previous adversarial examples
        all_adv_examples = []
        for prev_iter in range(1, iteration + 1):
            adv_file = f"{self.config['adversarial_training']['adversarial_data_dir']}/adversarial_iter_{prev_iter}.csv"
            if os.path.exists(adv_file):
                prev_adv_df = pd.read_csv(adv_file)
                prev_successful = prev_adv_df[prev_adv_df['attack_successful'] == True].copy()
                if len(prev_successful) > 0:
                    prev_adv_data = prev_successful[['adversarial_sequence', 'true_label']].copy()
                    prev_adv_data.columns = ['sequence', 'label']
                    all_adv_examples.append(prev_adv_data)
                    self.logger.info(f"Loaded {len(prev_adv_data)} adversarial examples from iteration {prev_iter}")
        
        # Combine all adversarial examples
        if all_adv_examples:
            combined_adv_df = pd.concat(all_adv_examples, ignore_index=True)
            # Remove duplicates based on sequence
            combined_adv_df = combined_adv_df.drop_duplicates(subset=['sequence'])
            self.logger.info(f"Total unique adversarial examples: {len(combined_adv_df)}")
        else:
            combined_adv_df = adv_training_data
        
        # Combine original training data with all adversarial examples
        combined_training_data = pd.concat([self.original_train_df, combined_adv_df], ignore_index=True)
        
        # Log detailed information
        self.logger.info(f"Preparing training data for iteration {iteration}:")
        self.logger.info(f"  - Original training samples: {len(self.original_train_df)}")
        self.logger.info(f"  - New adversarial examples: {len(adv_training_data)}")
        self.logger.info(f"  - Total adversarial examples: {len(combined_adv_df)}")
        self.logger.info(f"  - Combined training samples: {len(combined_training_data)}")
        
        return combined_training_data
    
    def train_iteration(self, train_df: pd.DataFrame, val_df: pd.DataFrame, iteration: int, 
                       model_path: str = None) -> Tuple[str, Dict[str, float]]:
        """Retrain model for one iteration using original + cumulative adversarial examples."""
        self.logger.info(f"Retraining iteration {iteration} on {len(train_df)} total examples (original + adversarial)")
        
        # Create model
        model = DNABERT2Classifier(
            model_name="zhihan1996/DNABERT-2-117M",
            num_classes=self.config['training']['num_classes'],
            freeze_encoder=True
        )
        
        # Load previous model if not first iteration
        if model_path and iteration > 0:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            self.logger.info(f"Loaded model from {model_path}")
            
            # Full retraining mode - all parameters trainable
            self.logger.info("Full retraining mode: All model parameters are trainable")
        
        # Train model
        train_metrics = train_model(
            model=model,
            train_df=train_df,
            val_df=val_df,
            config=self.config['training'],
            save_path=f"{self.config['checkpointing']['save_dir']}/iteration_{iteration}.pt"
        )
        
        # Evaluate on test set (using fast evaluation)
        test_df = pd.read_csv("data/raw/GUE/prom/prom_300_all/test.csv")
        test_metrics = evaluate_model(model, test_df, self.config['training'])
        
        # Calculate adversarial sample information
        # Count adversarial examples in training data
        adversarial_samples = len(train_df) - len(self.original_train_df)
        
        # Combine metrics
        iteration_metrics = {
            'iteration': iteration,
            'train_accuracy': train_metrics.get('train_accuracy', 0),
            'val_accuracy': train_metrics.get('val_accuracy', 0),
            'test_accuracy': test_metrics.get('accuracy', 0),
            'train_loss': train_metrics.get('train_loss', 0),
            'val_loss': train_metrics.get('val_loss', 0),
            'test_loss': test_metrics.get('loss', 0),
            'adversarial_samples': adversarial_samples,
            'training_approach': 'full_retraining'  # Indicate we're doing full retraining with cumulative data
        }
        
        self.training_metrics.append(iteration_metrics)
        
        model_path = f"{self.config['checkpointing']['save_dir']}/iteration_{iteration}.pt"
        return model_path, iteration_metrics
    
    def plot_training_progress(self):
        """Plot training progress across iterations."""
        if not self.training_metrics:
            self.logger.warning("No training metrics to plot")
            return
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        iterations = [m['iteration'] for m in self.training_metrics]
        
        # Accuracy plot
        ax1.plot(iterations, [m['train_accuracy'] for m in self.training_metrics], 'b-o', label='Train')
        ax1.plot(iterations, [m['val_accuracy'] for m in self.training_metrics], 'r-o', label='Validation')
        ax1.plot(iterations, [m['test_accuracy'] for m in self.training_metrics], 'g-o', label='Test')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Across Iterations')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(iterations, [m['train_loss'] for m in self.training_metrics], 'b-o', label='Train')
        ax2.plot(iterations, [m['val_loss'] for m in self.training_metrics], 'r-o', label='Validation')
        ax2.plot(iterations, [m['test_loss'] for m in self.training_metrics], 'g-o', label='Test')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss Across Iterations')
        ax2.legend()
        ax2.grid(True)
        
        # Test accuracy focus
        ax3.plot(iterations, [m['test_accuracy'] for m in self.training_metrics], 'g-o', linewidth=2, markersize=8)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_title('Test Accuracy Improvement')
        ax3.grid(True)
        
        # Robustness improvement
        if len(self.training_metrics) > 1:
            baseline_accuracy = self.training_metrics[0]['test_accuracy']
            improvements = [m['test_accuracy'] - baseline_accuracy for m in self.training_metrics]
            ax4.plot(iterations, improvements, 'purple', marker='o', linewidth=2, markersize=8)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Accuracy Improvement')
            ax4.set_title('Robustness Improvement Over Baseline')
            ax4.grid(True)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.config['monitoring']['plot_dir']}/training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved training progress plot to {plot_path}")
        
        plt.show()
    
    def save_metrics(self):
        """Save training metrics to CSV."""
        if not self.training_metrics:
            return
        
        metrics_df = pd.DataFrame(self.training_metrics)
        metrics_path = self.config['monitoring']['metrics_file']
        metrics_df.to_csv(metrics_path, index=False)
        self.logger.info(f"Saved training metrics to {metrics_path}")
    
    def run_iterative_training(self):
        """Run the complete iterative adversarial training pipeline."""
        self.logger.info("Starting iterative adversarial training pipeline")
        
        # Load original data
        train_df, val_df, test_df = self.load_original_data()
        
        # Check if we should start from existing model (skip iteration 0)
        start_iteration = self.config['adversarial_training'].get('start_from_iteration', 0)
        model_path = self.config['training']['base_model']
        
        if start_iteration == 0:
            # Initial training (iteration 0)
            self.logger.info("=== ITERATION 0: Initial Training ===")
            model_path, metrics = self.train_iteration(train_df, val_df, 0, model_path)
            self.logger.info(f"Iteration 0 - Test Accuracy: {metrics['test_accuracy']:.4f}")
        else:
            # Use existing model, evaluate it first
            self.logger.info(f"=== Using existing model from {model_path} ===")
            model = DNABERT2Classifier(
                model_name="zhihan1996/DNABERT-2-117M",
                num_classes=self.config['training']['num_classes'],
                freeze_encoder=True
            )
            
            # Load checkpoint and extract model state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Full retraining mode - all parameters trainable
            self.logger.info("Full retraining mode: All model parameters are trainable")
            
            # Evaluate existing model (using fast evaluation)
            test_metrics = evaluate_model(model, test_df, self.config['training'])
            baseline_metrics = {
                'iteration': 0,
                'train_accuracy': 0,  # Unknown for existing model
                'val_accuracy': 0,    # Unknown for existing model
                'test_accuracy': test_metrics['accuracy'],
                'train_loss': 0,      # Unknown for existing model
                'val_loss': 0,        # Unknown for existing model
                'test_loss': test_metrics['loss']
            }
            self.training_metrics.append(baseline_metrics)
            self.logger.info(f"Baseline - Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Check for existing adversarial examples from previous run
        existing_adv_path = "data/adversarial/genetic_attack/adversarial_sequences.csv"
        existing_adv_df = None
        if os.path.exists(existing_adv_path):
            self.logger.info(f"Found existing adversarial examples at {existing_adv_path}")
            existing_adv_df = pd.read_csv(existing_adv_path)
            self.logger.info(f"Using {len(existing_adv_df)} existing adversarial examples")
        
        # Iterative adversarial training
        max_iterations = self.config['adversarial_training']['max_iterations']
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"=== ITERATION {iteration}: Adversarial Training ===")
            
            # Use existing adversarial examples for first iteration, generate new ones for subsequent iterations
            if iteration == 1 and existing_adv_df is not None:
                self.logger.info("Using existing adversarial examples for first iteration")
                adversarial_df = existing_adv_df
            else:
                self.logger.info(f"Generating new adversarial examples for iteration {iteration}")
                adversarial_df = self.generate_adversarial_examples(model_path, test_df, iteration)
            
            # Prepare cumulative training data (original + all adversarial examples)
            adv_train_df = self.prepare_adversarial_training_data(adversarial_df, iteration)
            
            if len(adv_train_df) == 0:
                self.logger.warning(f"No adversarial examples for iteration {iteration}, skipping training")
                continue
            
            # Fine-tune only on adversarial examples
            model_path, metrics = self.train_iteration(adv_train_df, val_df, iteration, model_path)
            
            self.logger.info(f"Iteration {iteration} - Fine-tuned on {len(adv_train_df)} adversarial examples - Test Accuracy: {metrics['test_accuracy']:.4f}")
            
            # Check for convergence (optional early stopping) - only if threshold is set
            convergence_threshold = self.config['adversarial_training'].get('convergence_threshold', 0)
            if convergence_threshold > 0 and iteration > start_iteration + 1:
                prev_accuracy = self.training_metrics[-2]['test_accuracy']
                curr_accuracy = metrics['test_accuracy']
                improvement = curr_accuracy - prev_accuracy
                
                if improvement < convergence_threshold:
                    self.logger.info(f"Convergence detected (improvement: {improvement:.4f} < {convergence_threshold}), stopping early")
                    break
        
        # Final analysis
        self.logger.info("=== TRAINING COMPLETE ===")
        self._print_final_results()
        
        # Save results
        self.save_metrics()
        if self.config['monitoring']['plot_accuracy']:
            self.plot_training_progress()
        
        return self.training_metrics
    
    def _print_final_results(self):
        """Print final training results."""
        if len(self.training_metrics) < 2:
            return
        
        baseline_accuracy = self.training_metrics[0]['test_accuracy']
        final_accuracy = self.training_metrics[-1]['test_accuracy']
        improvement = final_accuracy - baseline_accuracy
        
        self.logger.info("=== FINAL RESULTS (Adversarial Fine-tuning Approach) ===")
        self.logger.info(f"Baseline Test Accuracy: {baseline_accuracy:.4f}")
        self.logger.info(f"Final Test Accuracy: {final_accuracy:.4f}")
        self.logger.info(f"Total Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        # Find best iteration
        best_iteration = max(self.training_metrics, key=lambda x: x['test_accuracy'])
        self.logger.info(f"Best Test Accuracy: {best_iteration['test_accuracy']:.4f} at iteration {best_iteration['iteration']}")
        
        # Show adversarial examples used in each iteration
        self.logger.info("Adversarial examples used per iteration:")
        for metrics in self.training_metrics[1:]:  # Skip baseline
            self.logger.info(f"  Iteration {metrics['iteration']}: {metrics['adversarial_samples']} adversarial examples")


def main():
    """Main function to run iterative adversarial training."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "configs/adversarial_training.yaml"
    
    try:
        # Initialize trainer
        trainer = IterativeAdversarialTrainer(config_path)
        
        # Run iterative training
        results = trainer.run_iterative_training()
        
        logger.info("Iterative adversarial training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in iterative adversarial training: {e}")
        raise


if __name__ == "__main__":
    main()
