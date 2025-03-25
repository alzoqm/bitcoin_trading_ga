from copy import deepcopy
import numpy as np
from ._base import BaseMutation
import torch


# class FlipSignMutation(BaseMutation):
#     """랜덤한 몇 부분의 +- 부호를 바꿔주는 mutation"""
    
#     def __init__(
#         self,
#         mut_prob: float = 0.05,
#     ):
#         self.mut_prob = mut_prob
        
#     def __call__(self, chromosome: np.ndarray) -> np.ndarray:
#         # mut_target = np.random.rand(*chromosome.shape) <= self.mut_prob
#         mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
#         mutant = deepcopy(chromosome)
#         mutant[mut_target] *= -1
#         return mutant
    
class FlipSignMutation(BaseMutation):
    """랜덤한 몇 부분의 +- 부호를 바꿔주는 mutation"""
    
    def __init__(
        self,
        mut_prob: float = 0.05,
    ):
        self.mut_prob = mut_prob
        
    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        mutant = chromosome.clone()
        mutant[mut_target] *= -1
        return mutant