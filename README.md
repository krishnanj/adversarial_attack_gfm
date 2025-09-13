# Adversarial Attacks on Gene Foundation Models

A research project investigating the adversarial robustness of DNABERT-2 and developing robust training methods for genomic sequence classification.

## Project Overview

This project explores the vulnerability of DNABERT-2 (a genomic foundation model) to adversarial attacks and develops iterative adversarial training methods to improve model robustness against minimal nucleotide perturbations.

### Key Research Questions

1. How vulnerable is DNABERT-2 to black-box adversarial attacks with minimal edits?
2. Can iterative adversarial training improve model robustness without sacrificing clean accuracy?
3. What are the biological implications of adversarial perturbations in genomic sequences?

## Repository Structure

```
adversarial_attack_gfm/
├── data/                    # Data storage
│   ├── raw/                # Raw datasets (FASTA/TXT files)
│   ├── processed/          # Processed CSV/TSV files
│   └── adversarial/        # Generated adversarial examples
├── models/                 # Model checkpoints
│   ├── baseline/          # Initial fine-tuned models
│   └── robust/            # Adversarially trained models
├── src/                   # Source code
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
├── reports/               # Generated plots and tables
├── scripts/               # Utility scripts
└── tests/                 # Unit tests
```

## Datasets

The project focuses on binary classification tasks:
- **Promoter vs Non-Promoter** sequences (~251bp)
- **Enhancer vs Non-Enhancer** sequences (~200bp)
- **Splice Site Prediction** (~60bp)

## Methodology

### Attack Strategy
- **Score-based attacks**: Greedy iterative nucleotide substitution
- **Decision-based attacks**: Hard-label black-box attacks using evolutionary algorithms
- **Biological constraints**: Preserve GC-content and regulatory motifs

### Training Pipeline
1. Fine-tune DNABERT-2 (frozen encoder) on clean data
2. Generate adversarial examples using black-box attacks
3. Iteratively retrain model on augmented adversarial data
4. Evaluate robustness improvements across multiple rounds

## Expected Outcomes

- Demonstrate DNABERT-2 vulnerability (85%+ attack success with 1-2 edits)
- Improve robust accuracy from ~10% to 50-70% through adversarial training
- Require 5-6 edits instead of 2 for successful attacks
- Maintain or improve clean accuracy

## Installation

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

*Detailed usage instructions will be added as the project develops.*

## Citation

If you use this work, please cite:

```bibtex
@article{adversarial_attack_gfm_2025,
  title={Adversarial Robustness in DNABERT-2 Sequence Classification},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DNABERT-2: Zhou et al. (2023) - ICLR 2024
- GenoArmory: Luo et al. (2025) - Adversarial evaluation benchmark
- TextAttack: Morris et al. (2020) - Adversarial attack framework