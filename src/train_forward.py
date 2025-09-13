#!/usr/bin/env python3
"""
DNABERT-2 fine-tuning script for genomic sequence classification.

This script implements the baseline training pipeline for DNABERT-2 on genomic
sequence classification tasks (promoter, enhancer, splice site prediction).
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_dataset, create_data_splits, load_dnabert2_model, 
    create_data_loaders, calculate_metrics, save_config, 
    load_config, set_seed, get_device
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DNABERT2Classifier(nn.Module):
    """
    DNABERT-2 classifier with frozen encoder and trainable classification head.
    """
    
    def __init__(self, model_name: str, num_classes: int = 2, freeze_encoder: bool = True):
        """
        Initialize the DNABERT-2 classifier.
        
        Args:
            model_name: Name of the DNABERT-2 model
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the DNABERT-2 encoder
        """
        super(DNABERT2Classifier, self).__init__()
        
        # Load DNABERT-2 model
        self.dnabert2 = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.dnabert2.parameters():
                param.requires_grad = False
            logger.info("DNABERT-2 encoder frozen")
        
        # Get hidden size from model config
        hidden_size = self.dnabert2.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        logger.info(f"Classifier initialized with {num_classes} classes")
        logger.info(f"Hidden size: {hidden_size}")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for the sequences
            
        Returns:
            Classification logits
        """
        # Get DNABERT-2 embeddings
        outputs = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        # DNABERT-2 returns a tuple, so we need to handle it properly
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state
        
        pooled_output = last_hidden_state[:, 0]  # [CLS] token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

class Trainer:
    """
    Trainer class for DNABERT-2 fine-tuning.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Create output directories
        self.model_dir = Path(config['output']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['output']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_model(self) -> None:
        """Set up the model, tokenizer, and optimizer."""
        logger.info("Setting up model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Initialize model
        self.model = DNABERT2Classifier(
            model_name=self.config['model']['name'],
            num_classes=self.config['model']['num_classes'],
            freeze_encoder=self.config['model']['freeze_encoder']
        )
        
        self.model.to(self.device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
        
    def setup_optimizer(self, num_training_steps: int) -> None:
        """Set up optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer...")
        
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Optimizer setup with {len(trainable_params)} parameter groups")
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate additional metrics
        metrics = calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save the model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Model saved to {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary of training history
        """
        logger.info("Starting training...")
        
        # Calculate total training steps
        num_training_steps = len(train_loader) * self.config['training']['num_epochs']
        self.setup_optimizer(num_training_steps)
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_accuracy = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val   - F1: {val_metrics['macro_avg_f1']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model(epoch, val_metrics)
                logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
        
        logger.info(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        
        return history

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DNABERT-2 on genomic sequences')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['promoter', 'enhancer', 'splice'],
                       help='Override dataset in config')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override dataset if specified
    if args.dataset:
        config['dataset']['name'] = args.dataset
        # Use GUE datasets
        if args.dataset == 'promoter':
            config['dataset']['path'] = "data/raw/GUE/prom/prom_300_all/train.csv"
            config['dataset']['max_length'] = 300
        elif args.dataset == 'enhancer':
            config['dataset']['path'] = "data/raw/GUE/EPI/GM12878/train.csv"
            config['dataset']['max_length'] = 200
        elif args.dataset == 'splice':
            config['dataset']['path'] = "data/raw/GUE/splice/reconstructed/train.csv"
            config['dataset']['max_length'] = 400
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    if config['device'] == 'auto':
        device = get_device()
    else:
        device = torch.device(config['device'])
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training on dataset: {config['dataset']['name']}")
    
    # Load dataset
    sequences, labels = load_dataset(config['dataset']['path'])
    
    # Create data splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        sequences, labels,
        test_size=config['data_splits']['test_size'],
        val_size=config['data_splits']['val_size'],
        random_state=config['data_splits']['random_state']
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['dataset']['max_length']
    )
    
    # Initialize trainer
    trainer = Trainer(config, device)
    trainer.setup_model()
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['macro_avg_f1']:.4f}")
    
    # Save final model
    trainer.save_model(config['training']['num_epochs'] - 1, test_metrics)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
