"""
Genetic Algorithm-based Adversarial Attack on DNABERT-2 using DEAP

This module implements a genetic algorithm approach using DEAP to generate adversarial
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
import yaml
from tqdm import tqdm

# DEAP for genetic algorithms
from deap import base, creator, tools, algorithms

# Bioinformatics libraries
from Bio.SeqUtils import gc_fraction
from Levenshtein import hamming

from transformers import AutoTokenizer, AutoModel
from utils import set_seed, load_config
from train_forward import DNABERT2Classifier


class BiologicalConstraints:
    """Handles biological plausibility constraints for genomic sequences using BioPython."""
    
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
        """Calculate GC content using BioPython."""
        return gc_fraction(sequence)  # BioPython returns fraction (0-1)
    
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
        """Calculate overall biological plausibility score based on enabled constraints."""
        scores = []
        
        # GC content constraint - only apply if max_gc_deviation is set
        if self.max_gc_deviation > 0:
            if self.gc_content_valid(original_seq, mutated_seq):
                scores.append(1.0)
            else:
                scores.append(0.0)
        
        # Motif preservation - only apply if preserve_motifs is enabled
        if self.preserve_motifs:
            scores.append(self.motif_preservation_score(original_seq, mutated_seq))
        
        # Transition preference - only apply if prefer_transitions is enabled
        if self.prefer_transitions and len(original_seq) == len(mutated_seq):
            transition_scores = []
            for orig, mut in zip(original_seq, mutated_seq):
                if orig != mut:
                    transition_scores.append(self.transition_preference(orig, mut))
            if transition_scores:
                scores.append(np.mean(transition_scores))
            else:
                scores.append(1.0)
        
        # If no constraints are enabled, return neutral score
        if not scores:
            return 1.0
        
        return np.mean(scores)


class GeneticAdversarialAttack:
    """Genetic algorithm-based adversarial attack on DNABERT-2 using DEAP."""
    
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
        
        # Setup DEAP framework
        self._setup_deap()
        
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
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def _load_tokenizer(self):
        """Load the DNABERT-2 tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.logger.info("Tokenizer loaded successfully")
    
    def _setup_deap(self):
        """Setup DEAP framework for genetic algorithm."""
        # Create fitness class (maximize fitness)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual class (list of positions to mutate)
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("mutate", self._deap_mutate)
        self.toolbox.register("mate", self._deap_crossover)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
    def _deap_mutate(self, individual):
        """DEAP mutation operator with adaptive mutation strategy."""
        # Adaptive mutation: more mutations for better exploration
        max_mutations = self.config['attack']['max_perturbations']
        num_mutations = random.randint(1, max_mutations)
        
        # Use different mutation strategies
        mutation_strategy = random.choice(['random', 'focused', 'aggressive'])
        
        if mutation_strategy == 'random':
            # Random mutations
            positions = random.sample(range(len(individual)), min(num_mutations, len(individual)))
            for pos in positions:
                individual[pos] = random.randint(0, 3)
                
        elif mutation_strategy == 'focused':
            # Focused mutations in a region
            start_pos = random.randint(0, len(individual) - num_mutations)
            for i in range(num_mutations):
                pos = start_pos + i
                if pos < len(individual):
                    individual[pos] = random.randint(0, 3)
                    
        else:  # aggressive
            # Aggressive mutations across the sequence
            positions = random.sample(range(len(individual)), min(num_mutations * 2, len(individual)))
            for pos in positions:
                individual[pos] = random.randint(0, 3)
        
        return (individual,)
    
    def _deap_crossover(self, ind1, ind2):
        """DEAP crossover operator."""
        # Single point crossover
        if len(ind1) > 1:
            point = random.randint(1, len(ind1) - 1)
            ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        return ind1, ind2
    
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
    
    def _calculate_fitness(self, original_seq: str, mutated_seq: str, 
                          original_confidence: float) -> Tuple[float]:
        """Calculate fitness score for a mutated sequence using DEAP format."""
        # Get model prediction for mutated sequence
        mutated_confidence, _ = self._predict_sequence(mutated_seq)
        
        # Calculate perturbations using Hamming distance
        perturbations = hamming(original_seq, mutated_seq)
        
        # Primary: confidence drop (we want to maximize this)
        confidence_drop = original_confidence - mutated_confidence
        
        # Secondary: perturbation penalty (we want to minimize perturbations)
        perturbation_penalty = perturbations * self.config['fitness']['perturbation_penalty']
        
        # Tertiary: biological penalty (we want high biological score)
        biological_score = self.bio_constraints.calculate_biological_score(original_seq, mutated_seq)
        biological_penalty = (1.0 - biological_score) * self.config['fitness']['biological_penalty']
        
        # Combined fitness (higher is better)
        # We want high confidence drop, low perturbations, high biological score
        fitness = (
            confidence_drop * self.config['fitness']['confidence_drop_weight'] -
            perturbation_penalty -
            biological_penalty
        )
        
        # Ensure original sequence (no perturbations) has lower fitness than mutated sequences
        if perturbations == 0:
            fitness = -1.0  # Original sequence should have negative fitness
        
        return (fitness,)
    
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
    
    
    def attack_sequence(self, sequence: str, true_label: int) -> Dict[str, Any]:
        """Perform genetic algorithm attack on a single sequence using DEAP."""
        self.logger.info(f"Attacking sequence of length {len(sequence)}")
        
        # Get original prediction
        original_confidence, original_prediction = self._predict_sequence(sequence)
        
        # Convert sequence to list of integers for DEAP
        nucleotides = ['A', 'T', 'C', 'G']
        nuc_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        int_to_nuc = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
        
        sequence_ints = [nuc_to_int[nuc] for nuc in sequence]
        
        # Create initial population
        population_size = self.config['genetic_algorithm']['population_size']
        population = []
        
        for _ in range(population_size):
            # Create individual with random mutations
            individual = sequence_ints.copy()
            
            # Vary mutation intensity for diversity
            mutation_intensity = random.choice(['light', 'medium', 'heavy'])
            if mutation_intensity == 'light':
                num_mutations = random.randint(1, 3)
            elif mutation_intensity == 'medium':
                num_mutations = random.randint(2, 5)
            else:  # heavy
                num_mutations = random.randint(4, self.config['attack']['max_perturbations'])
            
            positions = random.sample(range(len(individual)), min(num_mutations, len(individual)))
            
            for pos in positions:
                # Apply transition preference
                if self.bio_constraints.prefer_transitions:
                    transitions = {0: 3, 3: 0, 1: 2, 2: 1}  # A↔G, T↔C
                    if random.random() < self.bio_constraints.transition_prob:
                        individual[pos] = transitions.get(individual[pos], random.randint(0, 3))
                    else:
                        individual[pos] = random.randint(0, 3)
                else:
                    individual[pos] = random.randint(0, 3)
            
            # Create DEAP individual
            deap_individual = creator.Individual(individual)
            population.append(deap_individual)
        
        # Register evaluation function
        def evaluate(individual):
            mutated_seq = ''.join([int_to_nuc[i] for i in individual])
            return self._calculate_fitness(sequence, mutated_seq, original_confidence)
        
        self.toolbox.register("evaluate", evaluate)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop using DEAP
        best_fitness = -float('inf')
        no_improvement_count = 0
        convergence_threshold = self.config['genetic_algorithm']['convergence_threshold']
        
        for generation in range(self.config['genetic_algorithm']['max_generations']):
            # Select parents
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config['genetic_algorithm']['crossover_rate']:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.config['genetic_algorithm']['mutation_rate']:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Check for improvement
            current_best = max(population, key=lambda x: x.fitness.values[0])
            if current_best.fitness.values[0] > best_fitness:
                best_fitness = current_best.fitness.values[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= convergence_threshold:
                self.logger.info(f"Converged after {generation + 1} generations")
                break
        
        # Find best individual
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        best_sequence = ''.join([int_to_nuc[i] for i in best_individual])
        
        # Get final predictions
        adversarial_confidence, adversarial_prediction = self._predict_sequence(best_sequence)
        perturbations = hamming(sequence, best_sequence)
        biological_score = self.bio_constraints.calculate_biological_score(sequence, best_sequence)
        confidence_drop = original_confidence - adversarial_confidence
        
        # Check attack success
        success_criteria = self.config['success_criteria']
        attack_successful = (
            adversarial_confidence < success_criteria['confidence_threshold'] or
            confidence_drop >= success_criteria['min_confidence_drop']
        )
        
        return {
            'original_sequence': sequence,
            'original_confidence': original_confidence,
            'original_prediction': original_prediction,
            'true_label': true_label,
            'adversarial_sequence': best_sequence,
            'adversarial_confidence': adversarial_confidence,
            'adversarial_prediction': adversarial_prediction,
            'perturbations': perturbations,
            'biological_score': biological_score,
            'confidence_drop': confidence_drop,
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
            # Convert numpy objects to regular Python types for YAML
            clean_stats = self._clean_numpy_objects(attack_stats)
            stats_path = os.path.join(output_dir, 'attack_statistics.yaml')
            with open(stats_path, 'w') as f:
                yaml.dump(clean_stats, f, default_flow_style=False)
            self.logger.info(f"Attack statistics saved to {stats_path}")
    
    def run_attack(self, test_data):
        """Run the complete genetic algorithm attack."""
        self.logger.info("Starting genetic algorithm adversarial attack")
        
        # Load test data (either DataFrame or file path)
        if isinstance(test_data, str):
            test_data = pd.read_csv(test_data)
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
