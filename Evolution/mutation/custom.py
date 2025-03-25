import numpy as np
from copy import deepcopy
from ._base import BaseMutation

class GeneralMutation(BaseMutation):
    """General Mutation class that accepts mutation and distribution functions directly"""
    def __init__(
        self,
        mutation_func,  # Function to apply mutation
        distribution_func,  # Function to generate distribution
        loc: float = 1., 
        scale: float = 0.1, 
        mut_prob: float = 0.05,
    ):
        self.mutation_func = mutation_func
        self.distribution_func = distribution_func
        self.mut_prob = mut_prob

    def __call__(self, chromosome: np.ndarray) -> np.ndarray:
        factor = self.distribution_func(self.loc, self.scale, size=chromosome.shape).astype(chromosome.dtype)
        mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        return self.mutation_func(chromosome, factor, mut_target)

    def pick_by_rand(self, shape, prob):
        return np.random.rand(*shape) <= prob

# Example mutation functions
def mutation_add(chromosome, factor, target):
    chromosome[target] += factor[target]
    return chromosome

def mutation_multiply(chromosome, factor, target):
    chromosome[target] *= factor[target]
    return chromosome

# Example distribution functions
def distribution_normal(shape, loc=0., scale=0.1):
    return np.random.normal(loc, scale, size=shape).astype(np.float32)

def distribution_uniform(shape, low=-0.1, high=0.1):
    return np.random.uniform(low, high, size=shape).astype(np.float32)

# Named functions for specific distributions
def normal_add_distribution(shape):
    return distribution_normal(shape, loc=0., scale=0.1)

def uniform_add_distribution(shape):
    return distribution_uniform(shape, low=-0.1, high=0.1)

def normal_multiply_distribution(shape):
    return distribution_normal(shape, loc=1., scale=0.1)

def uniform_multiply_distribution(shape):
    return distribution_uniform(shape, low=0.8, high=1.2)

# Example usage
add_normal_mutation = GeneralMutation(
    mutation_func=mutation_add,
    distribution_func=normal_add_distribution,
    mut_prob=0.05
)

add_uniform_mutation = GeneralMutation(
    mutation_func=mutation_add,
    distribution_func=uniform_add_distribution,
    mut_prob=0.05
)

multiply_normal_mutation = GeneralMutation(
    mutation_func=mutation_multiply,
    distribution_func=normal_multiply_distribution,
    mut_prob=0.05
)

multiply_uniform_mutation = GeneralMutation(
    mutation_func=mutation_multiply,
    distribution_func=uniform_multiply_distribution,
    mut_prob=0.05
)