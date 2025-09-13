"""
Genetic Algorithm-based Adversarial Attack on DNABERT-2

This module implements a genetic algorithm approach to generate adversarial
sequences that maintain biological plausibility while fooling the model.
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import yaml
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from utils import set_seed, load_config
from train_forward import DNABERT2Classifier


@dataclass
class Individual:
    """Represents a single individual in the genetic algorithm population."""
    sequence: str
    fitness: float = 0.0
    perturbations: int = 0
    confidence_drop: float = 0.0
    biological_score: float = 1.0


class BiologicalConstraints:
    """Handles biological plausibility constraints for genomic sequences."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_gc_deviation = config['biological_constraints']['max_gc_deviation']
        self.preserve_motifs = config['biological_constraints']['preserve_motifs']
        self.prefer_transitions = config['biological_constraints']['prefer_transitions']
        self.avoid_stop_codons = config['biological_constraints']['avoid_stop_codons']
        
        # Common regulatory motifs to preserve
        self.important_motifs = [
            'TATA', 'CAAT', 'GC', 'AT', 'TTTT', 'AAAA'
        ]
        
        # Transition probabilities (A↔G, C↔T are more likely)
        self.transition_prob = 0.7 if self.prefer_transitions else 0.5
    
    def calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
    
    def gc_content_valid(self, original_seq: str, mutated_seq: str) -> bool:
        """Check if GC content deviation is within acceptable limits."""
        original_gc = self.calculate_gc_content(original_seq)
        mutated_gc = self.calculate_gc_content(mutated_seq)
        deviation = abs(original_gc - mutated_gc)
        return deviation <= self.max_gc_deviation
    
    def motif_preservation_score(self, original_seq: str, mutated_seq: str) -> float:
        """Calculate how well regulatory motifs are preserved."""
        if not self.preserve_motifs:
            return 1.0
        
        score = 0.0
        total_motifs = 0
        
        for motif in self.important_motifs:
            original_count = original_seq.count(motif)
            mutated_count = mutated_seq.count(motif)
            total_motifs += original_count
            
            if original_count > 0:
                preservation_ratio = min(mutated_count / original_count, 1.0)
                score += preservation_ratio * original_count
        
        return score / max(total_motifs, 1)
    
    def transition_preference(self, original_nuc: str, new_nuc: str) -> float:
        """Return preference score for transition vs transversion."""
        if not self.prefer_transitions:
            return 1.0
        
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        if (original_nuc, new_nuc) in transitions:
            return 1.0
        else:
            return 0.3  # Lower score for transversions
    
    def calculate_biological_score(self, original_seq: str, mutated_seq: str) -> float:
        """Calculate overall biological plausibility score."""
        scores = []
        
        # GC content constraint
        if self.gc_content_valid(original_seq, mutated_seq):
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Motif preservation
        scores.append(self.motif_preservation_score(original_seq, mutated_seq))
        
        # Transition preference (average across all changes)
        if len(original_seq) == len(mutated_seq):
            transition_scores = []
            for orig, mut in zip(original_seq, mutated_seq):
                if orig != mut:
                    transition_scores.append(self.transition_preference(orig, mut))
            if transition_scores:
                scores.append(np.mean(transition_scores))
            else:
                scores.append(1.0)
        
        return np.mean(scores)


class GeneticAdversarialAttack:
    """Genetic algorithm-based adversarial attack on DNABERT-2."""
    
    def __init__(self, config_path: str):
        """Initialize the genetic adversarial attack."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        set_seed(self.config.get('seed', 42))
        
        # Initialize biological constraints
        self.bio_constraints = BiologicalConstraints(self.config)
        
        # Load model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self._load_model()
        self._load_tokenizer()
        
        # Create output directory
        os.makedirs(self.config['output']['output_dir'], exist_ok=True)
    
    def _load_model(self):
        """Load the trained DNABERT-2 model."""
        model_path = self.config['attack']['target_model']
        self.logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DNABERT2Classifier(
            model_name="zhihan1996/DNABERT-2-117M",
            num_classes=2,
            freeze_encoder=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def _load_tokenizer(self):
        """Load the DNABERT-2 tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.logger.info("Tokenizer loaded successfully")
    
    def _mutate_sequence(self, sequence: str, num_mutations: int = 1) -> str:
        """Apply random mutations to a sequence."""
        nucleotides = ['A', 'T', 'C', 'G']
        mutated = list(sequence)
        
        # Select random positions to mutate
        positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
        
        for pos in positions:
            original_nuc = mutated[pos]
            # Choose new nucleotide (different from original)
            available_nucs = [n for n in nucleotides if n != original_nuc]
            
            # Apply transition preference
            if self.bio_constraints.prefer_transitions:
                transitions = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
                if random.random() < self.bio_constraints.transition_prob:
                    mutated[pos] = transitions.get(original_nuc, random.choice(available_nucs))
                else:
                    mutated[pos] = random.choice(available_nucs)
            else:
                mutated[pos] = random.choice(available_nucs)
        
        return ''.join(mutated)
    
    def _calculate_fitness(self, original_seq: str, individual: Individual, 
                          original_confidence: float) -> float:
        """Calculate fitness score for an individual."""
        # Primary: confidence drop (we want to maximize this)
        confidence_drop = original_confidence - individual.confidence_drop
        
        # Secondary: perturbation penalty (we want to minimize perturbations)
        perturbation_penalty = individual.perturbations * self.config['fitness']['perturbation_penalty']
        
        # Tertiary: biological penalty (we want high biological score)
        biological_penalty = (1.0 - individual.biological_score) * self.config['fitness']['biological_penalty']
        
        # Combined fitness (higher is better)
        # We want high confidence drop, low perturbations, high biological score
        fitness = (
            confidence_drop * self.config['fitness']['confidence_drop_weight'] -
            perturbation_penalty -
            biological_penalty
        )
        
        # Ensure original sequence (no perturbations) has lower fitness than mutated sequences
        if individual.perturbations == 0:
            fitness = -1.0  # Original sequence should have negative fitness
        
        return fitness
    
    def _predict_sequence(self, sequence: str) -> Tuple[float, int]:
        """Get model prediction for a sequence."""
        with torch.no_grad():
            # Tokenize sequence
            inputs = self.tokenizer(
                sequence,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.get('max_length', 300)
            )
            
            # Remove token_type_ids if present (DNABERT-2 doesn't use it)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            return confidence.item(), predicted_class.item()
    
    def _initialize_population(self, original_seq: str, original_confidence: float) -> List[Individual]:
        """Initialize the genetic algorithm population."""
        population = []
        population_size = self.config['genetic_algorithm']['population_size']
        
        # Add original sequence as first individual
        original_individual = Individual(
            sequence=original_seq,
            fitness=0.0,
            perturbations=0,
            confidence_drop=original_confidence,
            biological_score=1.0
        )
        population.append(original_individual)
        
        # Generate random variants
        for _ in range(population_size - 1):
            # Random number of mutations (1 to max_perturbations)
            num_mutations = random.randint(1, self.config['attack']['max_perturbations'])
            mutated_seq = self._mutate_sequence(original_seq, num_mutations)
            
            # Calculate properties
            confidence, _ = self._predict_sequence(mutated_seq)
            biological_score = self.bio_constraints.calculate_biological_score(original_seq, mutated_seq)
            
            individual = Individual(
                sequence=mutated_seq,
                perturbations=num_mutations,
                confidence_drop=confidence,
                biological_score=biological_score
            )
            
            population.append(individual)
        
        # Calculate fitness for all individuals
        for individual in population:
            individual.fitness = self._calculate_fitness(original_seq, individual, original_confidence)
        
        return population
    
    def _selection(self, population: List[Individual]) -> List[Individual]:
        """Select individuals for the next generation."""
        elite_ratio = self.config['genetic_algorithm']['elite_ratio']
        elite_count = int(len(population) * elite_ratio)
        
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Keep elite individuals
        elite = sorted_pop[:elite_count]
        
        # Tournament selection for the rest
        remaining_count = len(population) - elite_count
        selected = []
        
        for _ in range(remaining_count):
            # Tournament of size 3
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return elite + selected
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.config['genetic_algorithm']['crossover_rate']:
            return parent1, parent2
        
        seq1, seq2 = parent1.sequence, parent2.sequence
        if len(seq1) != len(seq2):
            return parent1, parent2
        
        # Single point crossover
        crossover_point = random.randint(1, len(seq1) - 1)
        
        child1_seq = seq1[:crossover_point] + seq2[crossover_point:]
        child2_seq = seq2[:crossover_point] + seq1[crossover_point:]
        
        # Create child individuals (fitness will be calculated later)
        child1 = Individual(sequence=child1_seq)
        child2 = Individual(sequence=child2_seq)
        
        return child1, child2
    
    def _mutate_individual(self, individual: Individual, original_seq: str) -> Individual:
        """Apply mutation to an individual."""
        if random.random() > self.config['genetic_algorithm']['mutation_rate']:
            return individual
        
        # Random number of mutations (1 to max_perturbations)
        num_mutations = random.randint(1, self.config['attack']['max_perturbations'])
        mutated_seq = self._mutate_sequence(individual.sequence, num_mutations)
        
        # Calculate new properties
        confidence, _ = self._predict_sequence(mutated_seq)
        biological_score = self.bio_constraints.calculate_biological_score(original_seq, mutated_seq)
        
        mutated_individual = Individual(
            sequence=mutated_seq,
            perturbations=num_mutations,
            confidence_drop=confidence,
            biological_score=biological_score
        )
        
        return mutated_individual
    
    def attack_sequence(self, sequence: str, true_label: int) -> Dict[str, Any]:
        """Perform genetic algorithm attack on a single sequence."""
        self.logger.info(f"Attacking sequence of length {len(sequence)}")
        
        # Get original prediction
        original_confidence, original_prediction = self._predict_sequence(sequence)
        
        # Initialize population
        population = self._initialize_population(sequence, original_confidence)
        
        # Evolution loop
        best_fitness = -float('inf')
        no_improvement_count = 0
        convergence_threshold = self.config['genetic_algorithm']['convergence_threshold']
        
        for generation in range(self.config['genetic_algorithm']['max_generations']):
            # Selection
            selected = self._selection(population)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    child1 = self._mutate_individual(child1, sequence)
                    child2 = self._mutate_individual(child2, sequence)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            # Calculate fitness for new individuals
            for individual in new_population:
                if individual.fitness == 0.0:  # New individual
                    individual.fitness = self._calculate_fitness(sequence, individual, original_confidence)
            
            # Update population
            population = new_population
            
            # Check for improvement
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= convergence_threshold:
                self.logger.info(f"Converged after {generation + 1} generations")
                break
        
        # Find best individual
        best_individual = max(population, key=lambda x: x.fitness)
        
        # Check attack success
        success_criteria = self.config['success_criteria']
        attack_successful = (
            best_individual.confidence_drop < success_criteria['confidence_threshold'] or
            (original_confidence - best_individual.confidence_drop) >= success_criteria['min_confidence_drop']
        )
        
        return {
            'original_sequence': sequence,
            'original_confidence': original_confidence,
            'original_prediction': original_prediction,
            'true_label': true_label,
            'adversarial_sequence': best_individual.sequence,
            'adversarial_confidence': best_individual.confidence_drop,
            'adversarial_prediction': self._predict_sequence(best_individual.sequence)[1],
            'perturbations': best_individual.perturbations,
            'biological_score': best_individual.biological_score,
            'confidence_drop': original_confidence - best_individual.confidence_drop,
            'attack_successful': attack_successful,
            'generations_used': generation + 1,
            'best_fitness': best_fitness
        }
    
    def attack_dataset(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Attack multiple sequences from the test dataset."""
        num_samples = min(self.config['attack']['test_samples'], len(test_data))
        test_samples = test_data.sample(n=num_samples, random_state=42)
        
        self.logger.info(f"Attacking {num_samples} sequences")
        
        results = []
        for idx, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Attacking sequences"):
            sequence = row['sequence']
            label = row['label']
            
            try:
                result = self.attack_sequence(sequence, label)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error attacking sequence {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df: pd.DataFrame, attack_stats: Dict[str, Any]):
        """Save attack results and statistics."""
        output_dir = self.config['output']['output_dir']
        
        # Save adversarial sequences
        if self.config['output']['save_adversarial_sequences']:
            sequences_path = os.path.join(output_dir, 'adversarial_sequences.csv')
            results_df.to_csv(sequences_path, index=False)
            self.logger.info(f"Adversarial sequences saved to {sequences_path}")
        
        # Save attack statistics
        if self.config['output']['save_attack_statistics']:
            stats_path = os.path.join(output_dir, 'attack_statistics.yaml')
            with open(stats_path, 'w') as f:
                yaml.dump(attack_stats, f, default_flow_style=False)
            self.logger.info(f"Attack statistics saved to {stats_path}")
    
    def run_attack(self, test_data_path: str):
        """Run the complete genetic algorithm attack."""
        self.logger.info("Starting genetic algorithm adversarial attack")
        
        # Load test data
        test_data = pd.read_csv(test_data_path)
        self.logger.info(f"Loaded {len(test_data)} test sequences")
        
        # Run attacks
        results_df = self.attack_dataset(test_data)
        
        # Calculate statistics
        if len(results_df) > 0:
            attack_stats = {
                'total_attacks': len(results_df),
                'successful_attacks': results_df['attack_successful'].sum(),
                'success_rate': results_df['attack_successful'].mean(),
                'avg_confidence_drop': results_df['confidence_drop'].mean(),
                'avg_perturbations': results_df['perturbations'].mean(),
                'avg_biological_score': results_df['biological_score'].mean(),
                'avg_generations': results_df['generations_used'].mean(),
                'config': self.config
            }
        else:
            attack_stats = {
                'total_attacks': 0,
                'successful_attacks': 0,
                'success_rate': 0.0,
                'avg_confidence_drop': 0.0,
                'avg_perturbations': 0.0,
                'avg_biological_score': 0.0,
                'avg_generations': 0.0,
                'config': self.config
            }
        
        # Save results
        self.save_results(results_df, attack_stats)
        
        # Print summary
        self.logger.info("Attack completed!")
        self.logger.info(f"Success rate: {attack_stats['success_rate']:.2%}")
        self.logger.info(f"Average confidence drop: {attack_stats['avg_confidence_drop']:.3f}")
        self.logger.info(f"Average perturbations: {attack_stats['avg_perturbations']:.1f}")
        
        return results_df, attack_stats


def main():
    """Main function to run genetic algorithm attack."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Genetic Algorithm Adversarial Attack')
    parser.add_argument('--config', type=str, default='configs/attack_genetic.yaml',
                       help='Path to attack configuration file')
    parser.add_argument('--test_data', type=str, 
                       default='data/raw/GUE/prom/prom_300_all/test.csv',
                       help='Path to test dataset')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run attack
    attack = GeneticAdversarialAttack(args.config)
    results_df, stats = attack.run_attack(args.test_data)
    
    print(f"\nAttack Results:")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence Drop: {stats['avg_confidence_drop']:.3f}")
    print(f"Average Perturbations: {stats['avg_perturbations']:.1f}")


if __name__ == "__main__":
    main()
