#!/usr/bin/env python3
"""
Utility functions for DNABERT-2 adversarial attack research.

This module provides data loading, model utilities, and helper functions
for training and evaluating DNABERT-2 models on genomic sequence classification.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenomicDataset(Dataset):
    """
    PyTorch Dataset for genomic sequence classification.
    
    Handles loading and tokenization of DNA sequences for DNABERT-2.
    """
    
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the genomic dataset.
        
        Args:
            sequences: List of DNA sequences
            labels: List of binary labels (0 or 1)
            tokenizer: DNABERT-2 tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Created dataset with {len(sequences)} sequences")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence and its label.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Dictionary containing tokenized sequence and label
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
    """
    Load a genomic dataset from CSV file.
    
    Args:
        dataset_path: Path to the CSV file containing sequences and labels
        
    Returns:
        Tuple of (sequences, labels)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Validate required columns
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'sequence' and 'label' columns")
    
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    
    # Validate sequences
    valid_bases = {'A', 'C', 'G', 'T'}
    for i, seq in enumerate(sequences):
        if not all(base in valid_bases for base in seq):
            logger.warning(f"Sequence {i} contains invalid bases: {seq}")
    
    logger.info(f"Loaded {len(sequences)} sequences")
    logger.info(f"Sequence lengths: min={min(len(s) for s in sequences)}, max={max(len(s) for s in sequences)}")
    
    return sequences, labels

def create_data_splits(sequences: List[str], labels: List[int], 
                      test_size: float = 0.2, val_size: float = 0.1,
                      random_state: int = 42) -> Tuple[List[str], List[str], List[str], 
                                                      List[int], List[int], List[int]]:
    """
    Create train/validation/test splits for the dataset.
    
    Args:
        sequences: List of DNA sequences
        labels: List of binary labels
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Creating train/validation/test splits...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data splits created:")
    logger.info(f"  Train: {len(X_train)} sequences")
    logger.info(f"  Validation: {len(X_val)} sequences")
    logger.info(f"  Test: {len(X_test)} sequences")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_dnabert2_model(model_name: str = "zhihan1996/DNABERT-2-117M") -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load DNABERT-2 model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name of the DNABERT-2 model to load
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading DNABERT-2 model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info("DNABERT-2 model loaded successfully")
        logger.info(f"Model config: {model.config}")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load DNABERT-2 model: {e}")
        raise

def create_data_loaders(X_train: List[str], X_val: List[str], X_test: List[str],
                       y_train: List[int], y_val: List[int], y_test: List[int],
                       tokenizer, batch_size: int = 32, max_length: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test sequences
        y_train, y_val, y_test: Training, validation, and test labels
        tokenizer: DNABERT-2 tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating data loaders...")
    
    # Create datasets
    train_dataset = GenomicDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = GenomicDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = GenomicDataset(X_test, y_test, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data loaders created with batch size {batch_size}")
    
    return train_loader, val_loader, test_loader

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_1': report['1']['f1-score'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }
    
    return metrics

def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")

def get_device() -> torch.device:
    """
    Get the appropriate device (GPU if available, otherwise CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device
