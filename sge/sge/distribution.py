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
    Single flattened Gaussian distribution for the entire genotype.

    This class maintains one mean vector and one std vector whose length is the
    total number of codons across all non-terminals (sum of max_expansions).
    The genotype is flattened when updating the distribution, and samples are
    split back into per-non-terminal arrays when creating genotypes.
    """

    def __init__(self, non_terminals: List[str], max_expansions: Dict[str, int],
                 init_std: float = 0.2, epsilon: float = 1e-8):
        self.non_terminals = non_terminals
        self.max_expansions = max_expansions
        self.sizes = [max_expansions[nt] for nt in non_terminals]
        self.cum_sizes = np.cumsum([0] + self.sizes)
        self.total_size = int(self.cum_sizes[-1])

        # Single flattened mean/std for the whole genotype
        self.means = np.random.uniform(0, 1, self.total_size)
        self.stds = np.full(self.total_size, init_std)
        self.epsilon = epsilon

    def sample_flat(self) -> np.ndarray:
        """
        Sample a flattened genotype vector from the distribution.

        Returns:
            1D numpy array of length `total_size` with values normalized via sigmoid to (0,1)
        """
        raw = np.random.normal(self.means, self.stds)
        return 1.0 / (1.0 + np.exp(-raw))

    def sample_genotype(self) -> List:
        """
        Convenience: sample and return a PSGE-style genotype split per non-terminal.
        Each codon is returned in the [-1, value, -1] triplet format.
        """
        flat = self.sample_flat()
        return create_genotype_from_samples(flat, self.non_terminals, self.max_expansions)

    def update(self, elite_individuals: List[Dict], n_best: int) -> None:
        """
        Update the single flattened distribution using the top `n_best` elites.

        Each individual's `genotype` is expected to be a list (per non-terminal)
        of arrays of codons in the format `[-1, value, -1]`. We flatten the
        middle values across all non-terminals for update.
        """
        elite = elite_individuals[:n_best]
        if len(elite) == 0:
            return

        # Build matrix of shape (n_elite, total_size)
        flat_matrix = np.zeros((len(elite), self.total_size), dtype=float)
        for i, ind in enumerate(elite):
            flat_vector = []
            for nt_idx, nt in enumerate(self.non_terminals):
                nt_gen = ind['genotype'][nt_idx]
                # extract middle values from [-1, value, -1]
                vals = [codon[1] for codon in nt_gen]
                flat_vector.extend(vals)
            flat_matrix[i, :] = np.array(flat_vector, dtype=float)

        # Update mean and std across elites
        self.means = np.mean(flat_matrix, axis=0)
        self.stds = np.std(flat_matrix, axis=0) + self.epsilon

    def get_state(self) -> Dict:
        return {'means': self.means.tolist(), 'stds': self.stds.tolist()}

    def set_state(self, state: Dict) -> None:
        means = np.asarray(state['means'], dtype=float)
        stds = np.asarray(state['stds'], dtype=float)
        expected_shape = (self.total_size,)
        if means.shape != expected_shape or stds.shape != expected_shape:
            raise ValueError(
                "Checkpoint genotype distribution shape does not match the current grammar"
            )
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
            raise ValueError("Checkpoint genotype distribution contains non-finite values")
        if np.any(stds < 0):
            raise ValueError("Checkpoint genotype distribution contains negative deviations")
        self.means = means.copy()
        self.stds = stds.copy()


def create_genotype_from_samples(samples: np.ndarray,
                                 non_terminals: List[str],
                                 max_expansions: Dict[str, int]) -> List:
    """
    Convert a flattened sample vector into PSGE genotype format split per NT.

    Args:
        samples: 1D numpy array of sampled values in [0,1], length = sum(max_expansions)
        non_terminals: List of non-terminal names in the same order used for flattening
        max_expansions: Dict mapping non-terminal to its number of codons

    Returns:
        Genotype as list of lists, where each codon is [-1, sampled_value, -1]
    """
    genotype = []
    idx = 0
    for nt in non_terminals:
        size = max_expansions[nt]
        segment = samples[idx:idx + size]
        nt_genotype = [[-1, float(val), -1] for val in segment]
        genotype.append(nt_genotype)
        idx += size
    return genotype
