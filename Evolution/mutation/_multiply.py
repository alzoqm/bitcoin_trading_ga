from copy import deepcopy
import numpy as np
from ._base import BaseMutation
import torch


# class MultiplyNormalMutation(BaseMutation):
#     """Normal Distribution에서 값을 뽑아 곱하는 mutation"""
    
#     def __init__(
#         self, 
#         loc: float = 1., 
#         scale: float = 0.1, 
#         mut_prob: float = 0.05,
#     ):
#         self.loc = loc
#         self.scale = scale
#         self.mut_prob = mut_prob
    
#     def __call__(self, chromosome: np.ndarray) -> np.ndarray:
        
#         mutiply_factor = np.random.normal(self.loc, self.scale, size=chromosome.shape).astype(chromosome.dtype)
#         # mut_target = np.random.rand(*chromosome.shape) <= self.mut_prob
#         mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        
#         # mutant = deepcopy(chromosome)
#         mutant = chromosome
#         mutant[mut_target] *= mutiply_factor[mut_target]
        
#         return mutant
    
class MultiplyNormalMutation(BaseMutation):
    """Normal Distribution에서 값을 뽑아 곱하는 mutation"""
    
    def __init__(
        self, 
        loc: float = 1., 
        scale: float = 0.1, 
        mut_prob: float = 0.05,
    ):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.mut_prob = mut_prob
    
    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        multiply_factor = torch.normal(self.loc, self.scale, size=chromosome.shape, device=chromosome.device)
        multiply_factor = multiply_factor.to(chromosome.dtype)
        
        mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        
        mutant = chromosome.clone()
        mutant[mut_target] *= multiply_factor[mut_target]
        
        return mutant
        

# class MultiplyUniformMutation(BaseMutation):
#     """Uniform Distribution에서 값을 뽑아 곱하는 mutation"""
    
#     def __init__(
#         self, 
#         low: float = 0.8, 
#         high: float = 1.2, 
#         mut_prob: float = 0.05,
#     ):
#         self.low = low
#         self.high = high
#         self.mut_prob = mut_prob
    
#     def __call__(self, chromosome: np.ndarray) -> np.ndarray:
        
#         mutiply_factor = np.random.uniform(self.low, self.high, size=chromosome.shape).astype(chromosome.dtype)
#         # mut_target = np.random.rand(*chromosome.shape) <= self.mut_prob
#         mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        
#         # mutant = deepcopy(chromosome)
#         mutant = chromosome
#         mutant[mut_target] *= mutiply_factor[mut_target]
        
#         return mutant
    
class MultiplyUniformMutation(BaseMutation):
    """Uniform Distribution에서 값을 뽑아 곱하는 mutation"""
    
    def __init__(
        self, 
        low: float = 0.8, 
        high: float = 1.2, 
        mut_prob: float = 0.05,
    ):
        self.low = low
        self.high = high
        self.mut_prob = mut_prob
    
    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        device = chromosome.device
        dtype = chromosome.dtype
        
        multiply_factor = torch.rand(chromosome.shape, device=device, dtype=dtype) * (self.high - self.low) + self.low
        mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        
        mutant = chromosome.clone()
        mutant[mut_target] *= multiply_factor[mut_target]
        
        return mutant
