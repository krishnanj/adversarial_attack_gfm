"""
Comprehensive plotting module for iterative adversarial training results.
Saves data files and generates publication-ready plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style for publication
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': False,  # Remove grid lines
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class AdversarialTrainingPlotter:
    """Generate comprehensive plots for iterative adversarial training results."""
    
    def __init__(self, results_dir="results/adversarial_training", plots_dir="plots/adversarial_training"):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.data_dir = self.results_dir / "plot_data"
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Plotter initialized - Results: {self.results_dir}, Plots: {self.plots_dir}")
    
    def load_training_metrics(self):
        """Load training metrics from CSV file."""
        # Check multiple possible locations
        possible_paths = [
            self.results_dir / "training_metrics.csv",
            Path("data/adversarial/iterative_training/training_metrics.csv"),
            Path("training_metrics.csv")
        ]
        
        metrics_file = None
        for path in possible_paths:
            if path.exists():
                metrics_file = path
                break
        
        if not metrics_file:
            raise FileNotFoundError(f"Training metrics not found in any of: {possible_paths}")
        
        df = pd.read_csv(metrics_file)
        logger.info(f"Loaded training metrics: {len(df)} iterations from {metrics_file}")
        return df
    
    def load_attack_statistics(self):
        """Load attack statistics from YAML files."""
        attack_stats = []
        
        # Load baseline attack stats (iteration 0)
        baseline_file = Path("data/adversarial/genetic_attack/attack_statistics.yaml")
        if baseline_file.exists():
            try:
                # Use yaml.load with FullLoader to handle numpy objects
                with open(baseline_file, 'r') as f:
                    stats = yaml.load(f, Loader=yaml.FullLoader)
                    if stats:
                        # Convert numpy objects to regular Python types
                        clean_stats = self._clean_numpy_objects(stats)
                        clean_stats['iteration'] = 0
                        attack_stats.append(clean_stats)
                        logger.info(f"Loaded baseline attack stats: {clean_stats}")
            except Exception as e:
                logger.warning(f"Could not load baseline stats: {e}")
        
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
                                clean_stats = self._clean_numpy_objects(stats)
                                clean_stats['iteration'] = i
                                attack_stats.append(clean_stats)
                                logger.info(f"Loaded iteration {i} attack stats: {clean_stats}")
                    except Exception as e:
                        logger.warning(f"Could not load iteration {i} stats: {e}")
        
        if not attack_stats:
            logger.warning("No attack statistics found")
            return pd.DataFrame()
        
        df = pd.DataFrame(attack_stats)
        logger.info(f"Loaded attack statistics: {len(df)} iterations")
        return df
    
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
    
    def save_plot_data(self, data_dict, filename):
        """Save plot data to CSV file."""
        filepath = self.data_dir / f"{filename}.csv"
        if isinstance(data_dict, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([data_dict])
        else:
            df = data_dict
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved plot data: {filepath}")
    
    def plot_model_robustness(self, training_df):
        """Plot 1: Model test accuracy vs iteration (robustness improvement)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot test accuracy
        ax.plot(training_df['iteration'], training_df['test_accuracy'], 
                'o-', linewidth=2, markersize=8, color='#2E86AB', label='Test Accuracy')
        
        # Add baseline reference line
        baseline_acc = training_df.iloc[0]['test_accuracy']
        ax.axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.7, 
                  label=f'Baseline ({baseline_acc:.3f})')
        
        ax.set_xlabel('Adversarial Training Iteration')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Model Robustness: Test Accuracy vs Iteration')
        ax.legend()
        ax.set_ylim(0.8, 1.0)
        
        # Save data
        self.save_plot_data(training_df[['iteration', 'test_accuracy']], 'model_robustness')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_robustness.png')
        plt.savefig(self.plots_dir / 'model_robustness.pdf')
        plt.close()
        logger.info("Generated model robustness plot")
    
    def plot_attack_effectiveness(self, attack_df):
        """Plot 2: Attack success rate vs iteration (attack difficulty)."""
        if attack_df.empty:
            logger.warning("No attack data available for effectiveness plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot success rate
        ax.plot(attack_df['iteration'], attack_df['success_rate'], 
                'o-', linewidth=2, markersize=8, color='#A23B72', label='Attack Success Rate')
        
        ax.set_xlabel('Adversarial Training Iteration')
        ax.set_ylabel('Attack Success Rate')
        ax.set_title('Attack Effectiveness: Success Rate vs Iteration')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Save data
        self.save_plot_data(attack_df[['iteration', 'success_rate']], 'attack_effectiveness')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attack_effectiveness.png')
        plt.savefig(self.plots_dir / 'attack_effectiveness.pdf')
        plt.close()
        logger.info("Generated attack effectiveness plot")
    
    def plot_attack_difficulty(self, attack_df):
        """Plot 3: Average confidence drop vs iteration."""
        if attack_df.empty:
            logger.warning("No attack data available for difficulty plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot confidence drop
        ax.plot(attack_df['iteration'], attack_df['avg_confidence_drop'], 
                'o-', linewidth=2, markersize=8, color='#F18F01', label='Avg Confidence Drop')
        
        ax.set_xlabel('Adversarial Training Iteration')
        ax.set_ylabel('Average Confidence Drop')
        ax.set_title('Attack Difficulty: Confidence Drop vs Iteration')
        ax.legend()
        
        # Save data
        self.save_plot_data(attack_df[['iteration', 'avg_confidence_drop']], 'attack_difficulty')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attack_difficulty.png')
        plt.savefig(self.plots_dir / 'attack_difficulty.pdf')
        plt.close()
        logger.info("Generated attack difficulty plot")
    
    def plot_perturbation_efficiency(self, attack_df):
        """Plot 4: Average perturbations needed vs iteration."""
        if attack_df.empty:
            logger.warning("No attack data available for perturbation plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot perturbations
        ax.plot(attack_df['iteration'], attack_df['avg_perturbations'], 
                'o-', linewidth=2, markersize=8, color='#C73E1D', label='Avg Perturbations')
        
        ax.set_xlabel('Adversarial Training Iteration')
        ax.set_ylabel('Average Perturbations')
        ax.set_title('Perturbation Efficiency: Required Changes vs Iteration')
        ax.legend()
        
        # Save data
        self.save_plot_data(attack_df[['iteration', 'avg_perturbations']], 'perturbation_efficiency')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'perturbation_efficiency.png')
        plt.savefig(self.plots_dir / 'perturbation_efficiency.pdf')
        plt.close()
        logger.info("Generated perturbation efficiency plot")
    
    def plot_biological_plausibility(self, attack_df):
        """Plot 5: Average biological score vs iteration."""
        if attack_df.empty or 'avg_biological_score' not in attack_df.columns:
            logger.warning("No biological score data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot biological score
        ax.plot(attack_df['iteration'], attack_df['avg_biological_score'], 
                'o-', linewidth=2, markersize=8, color='#2D5016', label='Avg Biological Score')
        
        ax.set_xlabel('Adversarial Training Iteration')
        ax.set_ylabel('Average Biological Score')
        ax.set_title('Biological Plausibility: Score vs Iteration')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Save data
        self.save_plot_data(attack_df[['iteration', 'avg_biological_score']], 'biological_plausibility')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'biological_plausibility.png')
        plt.savefig(self.plots_dir / 'biological_plausibility.pdf')
        plt.close()
        logger.info("Generated biological plausibility plot")
    
    def plot_training_progress(self, training_df):
        """Plot 6: Training and validation loss vs iteration."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot losses
        ax.plot(training_df['iteration'], training_df['train_loss'], 
                'o-', linewidth=2, markersize=8, color='#4A90E2', label='Training Loss')
        ax.plot(training_df['iteration'], training_df['val_loss'], 
                's-', linewidth=2, markersize=8, color='#7ED321', label='Validation Loss')
        
        ax.set_xlabel('Adversarial Training Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress: Loss vs Iteration')
        ax.legend()
        
        # Save data
        loss_data = training_df[['iteration', 'train_loss', 'val_loss']]
        self.save_plot_data(loss_data, 'training_progress')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_progress.png')
        plt.savefig(self.plots_dir / 'training_progress.pdf')
        plt.close()
        logger.info("Generated training progress plot")
    
    def plot_comprehensive_summary(self, training_df, attack_df):
        """Plot 7: Comprehensive summary with multiple metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Model robustness
        ax1.plot(training_df['iteration'], training_df['test_accuracy'], 
                'o-', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_title('Model Robustness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_ylim(0.8, 1.0)
        
        # Subplot 2: Attack effectiveness
        if not attack_df.empty:
            ax2.plot(attack_df['iteration'], attack_df['success_rate'], 
                    'o-', linewidth=2, markersize=6, color='#A23B72')
            ax2.set_title('Attack Effectiveness')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Success Rate')
            ax2.set_ylim(0, 1)
        
        # Subplot 3: Attack difficulty
        if not attack_df.empty:
            ax3.plot(attack_df['iteration'], attack_df['avg_confidence_drop'], 
                    'o-', linewidth=2, markersize=6, color='#F18F01')
            ax3.set_title('Attack Difficulty')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Confidence Drop')
        
        # Subplot 4: Perturbation efficiency
        if not attack_df.empty:
            ax4.plot(attack_df['iteration'], attack_df['avg_perturbations'], 
                    'o-', linewidth=2, markersize=6, color='#C73E1D')
            ax4.set_title('Perturbation Efficiency')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Avg Perturbations')
        
        plt.suptitle('Iterative Adversarial Training: Comprehensive Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'comprehensive_summary.png')
        plt.savefig(self.plots_dir / 'comprehensive_summary.pdf')
        plt.close()
        logger.info("Generated comprehensive summary plot")
    
    def generate_all_plots(self):
        """Generate all plots and save data files."""
        logger.info("Starting comprehensive plot generation...")
        
        # Load data
        training_df = self.load_training_metrics()
        attack_df = self.load_attack_statistics()
        
        # Generate individual plots
        self.plot_model_robustness(training_df)
        self.plot_attack_effectiveness(attack_df)
        self.plot_attack_difficulty(attack_df)
        self.plot_perturbation_efficiency(attack_df)
        self.plot_biological_plausibility(attack_df)
        self.plot_training_progress(training_df)
        self.plot_comprehensive_summary(training_df, attack_df)
        
        logger.info(f"All plots generated and saved to: {self.plots_dir}")
        logger.info(f"Plot data saved to: {self.data_dir}")

def main():
    """Main function to generate all plots."""
    plotter = AdversarialTrainingPlotter()
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()
