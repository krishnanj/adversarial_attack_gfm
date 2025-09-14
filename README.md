# Adversarial Attacks on Gene Foundation Models

A comprehensive framework for investigating and improving the adversarial robustness of DNABERT-2 through iterative adversarial training on genomic sequence classification tasks.

## Overview

This project implements an iterative adversarial training pipeline that enhances the robustness of DNABERT-2 against genetic algorithm-based adversarial attacks. The framework supports both promoter and transcription factor classification tasks with configurable training parameters.

## Key Features

- **Multi-Dataset Support**: Works with both promoter (300bp) and transcription factor (200bp) datasets
- **Iterative Adversarial Training**: Fine-tunes models exclusively on adversarial examples
- **Biological Constraints**: Maintains biological plausibility in adversarial generation
- **Configurable Parameters**: Full control over training and attack parameters via YAML configs
- **Lambda Labs Integration**: Complete setup guide for cloud GPU experiments
- **Comprehensive Logging**: Detailed progress tracking and result visualization

## Quick Start

### Local Setup
```bash
# Clone repository
git clone https://github.com/krishnanj/adversarial_attack_gfm.git
cd adversarial_attack_gfm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fix DNABERT-2 compatibility (CRITICAL)
pip install cmake
git clone https://github.com/openai/triton.git
cd triton
pip install -e .
pip uninstall triton
cd ~/adversarial_attack_gfm
```

### Run Experiments

**Transcription Factor Dataset (Recommended)**:
```bash
# Full training (5 iterations, ~2-3 hours)
python scripts/run_adversarial_training.py --dataset tf

# Quick testing (2-3 minutes)
# First modify configs for fast parameters, then:
python scripts/run_adversarial_training.py --dataset tf
```

**Promoter Dataset**:
```bash
python scripts/run_adversarial_training.py --dataset promoter
```

## Cloud GPU Setup

For running experiments on Lambda Labs GPU instances, see the complete setup guide:

**[lambda_labs_setup.md](lambda_labs_setup.md)**

The guide includes:
- Instance setup and SSH configuration
- DNABERT-2 compatibility fixes
- Background execution commands
- Cost optimization tips
- Troubleshooting common issues

## Dataset Support

### Transcription Factor Dataset
- **Path**: `data/raw/GUE/tf/0/`
- **Sequences**: 32,378 training samples
- **Length**: 200bp sequences
- **Task**: Binary classification of transcription factor binding sites

### Promoter Dataset
- **Path**: `data/raw/GUE/prom/prom_300_all/`
- **Sequences**: 47,356 training samples
- **Length**: 300bp sequences
- **Task**: Binary classification of promoter vs non-promoter sequences

## Configuration

### Training Parameters
Edit `configs/adversarial_training.yaml`:
```yaml
training:
  dataset: "tf"  # or "promoter"
  num_epochs: 3
  batch_size: 16
  max_iterations: 5
  convergence_threshold: 0  # No early stopping
```

### Attack Parameters
Edit `configs/attack_genetic.yaml`:
```yaml
attack:
  test_samples: 50
  max_perturbations: 8
genetic_algorithm:
  population_size: 100
  max_generations: 200
```

## Expected Results

### Full Training (5 iterations)
- **Runtime**: 2-3 hours
- **Improvement**: Model becomes more robust to adversarial attacks
- **Output**: Comprehensive plots and metrics in `results/adversarial_training/`

### Quick Testing
- **Runtime**: 2-3 minutes
- **Purpose**: Verify setup and functionality
- **Output**: Basic results for validation

## Project Structure

```
adversarial_attack_gfm/
├── src/                          # Core implementation
│   ├── adversarial_training.py   # Main training pipeline
│   ├── attack_genetic.py         # Genetic algorithm attacks
│   ├── train_forward.py          # Model training utilities
│   └── utils.py                  # Common utilities
├── scripts/                      # Entry point scripts
│   └── run_adversarial_training.py
├── configs/                      # Configuration files
│   ├── adversarial_training.yaml
│   └── attack_genetic.yaml
├── data/                         # Datasets (excluded from git)
├── results/                      # Experiment results
├── plots/                        # Generated visualizations
└── lambda_labs_setup.md         # Cloud GPU setup guide
```

## Key Technical Details

### DNABERT-2 Compatibility
The project includes a critical fix for DNABERT-2 compatibility issues with newer Triton versions. The setup process installs Triton from source then removes it, allowing DNABERT-2 to fall back to standard PyTorch attention.

### Iterative Training Approach
Unlike traditional adversarial training, this framework:
1. Trains on original dataset
2. Generates adversarial examples against current model
3. Fine-tunes exclusively on adversarial examples
4. Repeats for specified iterations
5. Preserves pre-trained encoder throughout

### Biological Constraints
Adversarial generation includes:
- GC content preservation (max 5% deviation)
- Motif preservation
- Transition mutation preference (A↔G, C↔T)
- Stop codon avoidance

## Troubleshooting

### Common Issues

**Triton Compatibility Error**:
```
TypeError: dot() got an unexpected keyword argument 'trans_b'
```
**Solution**: Follow the Triton install/remove process in the setup instructions.

**Early Stopping**:
The config now has `convergence_threshold: 0` to ensure all iterations run as requested.

**GPU Memory Issues**:
Reduce `batch_size` in config files or use gradient accumulation.

## Research Applications

This framework is designed for:
- Adversarial robustness research in genomics
- Model security evaluation
- Robust training method development
- Biological sequence analysis

## Citation

If you use this work, please cite:

```bibtex
@article{adversarial_attack_gfm_2025,
  title={Adversarial Attacks on Gene Foundation Models: Iterative Adversarial Training for Robust Genomic Sequence Classification},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2025},
  note={Code available at: https://github.com/krishnanj/adversarial_attack_gfm}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **DNABERT-2**: Zhou et al. (2023) - ICLR 2024
- **GUE Benchmark**: Genomic Understanding Evaluation dataset
- **DEAP**: Distributed Evolutionary Algorithms in Python
- **BioPython**: Bioinformatics tools for Python
- **HuggingFace Transformers**: Model implementation and tokenization