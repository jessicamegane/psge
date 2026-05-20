"""
CMA-ES-inspired Probabilistic Sampling for PSGE Genotypes

This module implements a learned Gaussian distribution mechanism for PSGE genotypes.
Instead of uniform random initialization, each non-terminal learns a Gaussian
distribution (mean + diagonal covariance) that evolves based on elite individuals.

Key design decisions:
- Distributions are learned independently per non-terminal to respect grammar structure
- Diagonal covariance only (mean + std per gene), avoiding full covariance complexity
- Values are NOT probabilities; they are continuous genotype codons in [0,1]
- Clipping to [0,1] ensures valid allele ranges while preserving learning signal
- This structure allows future extension to full covariance / CMA-ES evolution paths
"""

import numpy as np
from typing import List, Dict, Tuple


class DiagonalGaussianDistribution:
    """
    Manages per-non-terminal Gaussian distributions with diagonal covariance.
    
    Attributes:
        means: Dict[int, np.ndarray] - Mean vector for each non-terminal
        stds: Dict[int, np.ndarray] - Standard deviation vector for each non-terminal
        epsilon: float - Small constant to prevent std from becoming zero
    """
    
    def __init__(self, non_terminals: List[str], max_expansions: Dict[str, int], 
                 init_std: float = 0.2, epsilon: float = 1e-8):
        """
        Initialize distributions for all non-terminals.
        
        Args:
            non_terminals: List of non-terminal names/indices
            max_expansions: Dict mapping non-terminal to max expansion count
            init_std: Initial standard deviation for all genes (default 0.2)
            epsilon: Small constant to prevent std collapse (default 1e-8)
        """
        self.means = {}
        self.stds = {}
        self.epsilon = epsilon
        self.non_terminals = non_terminals
        self.max_expansions = max_expansions
        
        # Initialize distributions for each non-terminal
        for nt in non_terminals:
            size = max_expansions[nt]
            # Means sampled uniformly from [0, 1]
            self.means[nt] = np.random.uniform(0, 1, size)
            # Stds initialized to constant value
            self.stds[nt] = np.full(size, init_std)
    
    def sample(self) -> Dict[int, np.ndarray]:
        """
        Sample genotype values from learned Gaussian distributions.
        
        Returns:
            Dict mapping non-terminal to sampled genotype array (clipped to [0,1])
            
        Why clipping is necessary:
        - Gaussian can produce values outside [0,1]
        - Clipping preserves the learning signal (mean/std) while enforcing domain
        - Values that land outside [0,1] are reflected back, not rejected
        """
        samples = {}
        for nt in self.non_terminals:
            # Sample from Gaussian: N(mean, std^2)
            raw_samples = np.random.normal(self.means[nt], self.stds[nt])
            # Clip to valid genotype range [0, 1]
            samples[nt] = np.clip(raw_samples, 0, 1)
        return samples
    
    def update(self, elite_individuals: List[Dict], n_best: int) -> None:
        """
        Update distributions based on elite individuals.
        
        For each non-terminal:
        - mean := average of elite genotype values
        - std := standard deviation of elite genotype values + epsilon
        
        Args:
            elite_individuals: Population individuals (assumed sorted by fitness)
            n_best: Number of elite individuals to use for update
            
        Why per-non-terminal update:
        - Each NT in the grammar has different expansion patterns
        - Independent learning preserves structural properties of the grammar
        - Allows future context-aware updates (e.g., depth-based distributions)
        """
        elite = elite_individuals[:n_best]
        
        for nt_idx, nt in enumerate(self.non_terminals):
            if len(elite) == 0:
                continue
            
            # Gather all genotype values for this NT from elite individuals
            elite_values = np.array([ind['genotype'][nt_idx] for ind in elite])
            
            # Extract the actual sampled values (middle element of [-1, value, -1] triplets)
            # Shape: [n_elite, max_expansions[nt]]
            codon_values = elite_values[:, :, 1].astype(float)
            
            # Update mean as average across elites
            self.means[nt_idx] = np.mean(codon_values, axis=0)
            
            # Update std as sample standard deviation + epsilon
            # epsilon prevents std from collapsing to zero
            self.stds[nt_idx] = np.std(codon_values, axis=0) + self.epsilon
    
    def get_state(self) -> Dict:
        """
        Get current distribution state for logging/checkpointing.
        
        Returns:
            Dict with means and stds for all non-terminals
        """
        return {
            'means': {nt: self.means[nt].tolist() for nt in self.non_terminals},
            'stds': {nt: self.stds[nt].tolist() for nt in self.non_terminals}
        }


def create_genotype_from_samples(samples: Dict[int, np.ndarray], 
                                 non_terminals: List[str]) -> List:
    """
    Convert distribution samples into PSGE genotype format.
    
    Args:
        samples: Dict mapping non-terminal to sampled values in [0,1]
        non_terminals: List of non-terminal indices
        
    Returns:
        Genotype as list of lists, where each codon is [-1, sampled_value, -1]
        
    Note:
        The [-1, value, -1] format is used by PSGE mapping logic.
        -1 is a placeholder; only the middle value (sampled codon) is used.
    """
    genotype = []
    for nt in non_terminals:
        nt_genotype = [[-1, sample, -1] for sample in samples[nt]]
        genotype.append(nt_genotype)
    return genotype
