#!/usr/bin/env python3
"""
Lambda Labs Experiment Runner
This script runs the adversarial training experiment on Lambda Labs GPU instances.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import yaml
import logging

def setup_logging():
    """Setup logging for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('lambda_labs_experiment.log')
        ]
    )
    return logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available and get GPU info."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ GPU available!")
            logger.info(f"GPU Info:\n{result.stdout}")
            return True
        else:
            logger.warning("⚠️ GPU not available, running on CPU")
            return False
    except FileNotFoundError:
        logger.warning("⚠️ nvidia-smi not found, running on CPU")
        return False

def check_disk_space():
    """Check available disk space."""
    try:
        result = subprocess.run(['df', '-h', '/tmp'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Disk space:\n{result.stdout}")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")

def setup_environment():
    """Setup environment variables for GPU training."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set Lambda Labs specific environment
    os.environ['LAMBDA_LABS'] = '1'
    
    logger.info("🌍 Environment variables set for GPU training")

def run_quick_experiment():
    """Run the quick version of the experiment."""
    logger.info("🚀 Starting QUICK experiment (fast parameters)")
    
    # Use the existing quick config
    config_file = "configs/adversarial_training.yaml"
    
    if not Path(config_file).exists():
        logger.error(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        # Run the experiment
        cmd = [sys.executable, "scripts/run_adversarial_training.py"]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        logger.info(f"⏱️ Experiment completed in {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            logger.info("✅ Quick experiment completed successfully!")
            logger.info(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ Quick experiment failed with return code {result.returncode}")
            logger.error(f"Error:\n{result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running quick experiment: {e}")
        return False

def run_full_experiment():
    """Run the full version of the experiment (using modified quick config)."""
    logger.info("🚀 Starting FULL experiment (using quick config with extended parameters)")
    
    # For now, we'll use the quick config but run for more iterations
    # You can modify the config file directly or create a new one later
    config_file = "configs/adversarial_training.yaml"
    
    if not Path(config_file).exists():
        logger.error(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        # Run the experiment with the existing config
        cmd = [sys.executable, "scripts/run_adversarial_training.py"]
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("Note: Modify configs/adversarial_training.yaml to increase max_iterations for full experiment")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        logger.info(f"⏱️ Experiment completed in {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            logger.info("✅ Full experiment completed successfully!")
            logger.info(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ Full experiment failed with return code {result.returncode}")
            logger.error(f"Error:\n{result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running full experiment: {e}")
        return False

def generate_plots():
    """Generate plots from the experiment results."""
    logger.info("📊 Generating plots...")
    
    try:
        cmd = [sys.executable, "scripts/generate_plots.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Plots generated successfully!")
            return True
        else:
            logger.error(f"❌ Plot generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating plots: {e}")
        return False

def save_results():
    """Save experiment results and logs."""
    logger.info("💾 Saving experiment results...")
    
    # Create results directory
    results_dir = Path("results/lambda_labs_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy important files
    important_files = [
        "data/adversarial/iterative_training/training_metrics.csv",
        "data/adversarial/iterative_training/",
        "plots/adversarial_training/",
        "models/adversarial_training/",
        "lambda_labs_experiment.log"
    ]
    
    for file_path in important_files:
        src = Path(file_path)
        if src.exists():
            if src.is_file():
                dst = results_dir / src.name
                subprocess.run(['cp', str(src), str(dst)])
                logger.info(f"📄 Copied {src} to {dst}")
            elif src.is_dir():
                dst = results_dir / src.name
                subprocess.run(['cp', '-r', str(src), str(dst)])
                logger.info(f"📁 Copied {src} to {dst}")
    
    logger.info(f"✅ Results saved to {results_dir}")

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run adversarial training experiment on Lambda Labs")
    parser.add_argument("--mode", choices=["quick", "full", "both"], default="quick",
                       help="Experiment mode: quick (fast params), full (complete params), or both")
    parser.add_argument("--skip-setup", action="store_true",
                       help="Skip environment setup (if already done)")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    logger.info("🚀 Lambda Labs Adversarial Training Experiment")
    logger.info(f"Mode: {args.mode}")
    
    # Check system status
    logger.info("🔍 Checking system status...")
    gpu_available = check_gpu_availability()
    check_disk_space()
    
    # Setup environment
    if not args.skip_setup:
        setup_environment()
    
    # Run experiments
    success = True
    
    if args.mode in ["quick", "both"]:
        logger.info("=" * 50)
        logger.info("RUNNING QUICK EXPERIMENT")
        logger.info("=" * 50)
        success &= run_quick_experiment()
    
    if args.mode in ["full", "both"] and success:
        logger.info("=" * 50)
        logger.info("RUNNING FULL EXPERIMENT")
        logger.info("=" * 50)
        success &= run_full_experiment()
    
    # Generate plots
    if success:
        logger.info("=" * 50)
        logger.info("GENERATING PLOTS")
        logger.info("=" * 50)
        generate_plots()
    
    # Save results
    save_results()
    
    if success:
        logger.info("🎉 All experiments completed successfully!")
        logger.info("📊 Check results/ directory for outputs")
        logger.info("💡 Remember to terminate your Lambda Labs instance!")
    else:
        logger.error("❌ Some experiments failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
