from copy import deepcopy
import numpy as np
from ._base import BaseMutation
import torch
import random



class AddNormalMutation(BaseMutation):
    """Normal Distribution에서 값을 뽑아 더하는 mutation"""
    def __init__(
        self,
        loc: float = 0.,
        scale: float = 0.1,
        mut_prob: float = 0.05,
    ):
        self.loc = loc
        self.scale = scale
        self.mut_prob = mut_prob

    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        # print(chromosome.shape)
        add_factor = torch.normal(self.loc, self.scale, size=chromosome.shape, dtype=chromosome.dtype)
        mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        mutant = chromosome.clone()  # Use clone to avoid in-place modification
        mutant[mut_target] += add_factor[mut_target]
        return mutant
    
class RandomValueMutation(BaseMutation):
    """일정 확률로 일부 값을 무작위 값으로 변경하는 mutation"""
    def __init__(
        self,
        min_val: float = -3.0,
        max_val: float = 3.0,
        mut_prob: float = 0.05,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.mut_prob = mut_prob

    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        # 변이될 위치 선택
        mut_target = self.pick_by_rand(chromosome.shape, self.mut_prob)
        # 무작위 값 생성
        random_values = torch.empty(chromosome.shape, dtype=chromosome.dtype).uniform_(self.min_val, self.max_val)
        # 원본을 복제하여 변이 적용
        mutant = chromosome.clone()
        mutant[mut_target] = random_values[mut_target]
        return mutant
        
class AddUniformMutation(BaseMutation):
    """Uniform Distribution에서 값을 뽑아 더하는 mutation"""
    def __init__(
        self,
        low: float = -0.1,
        high: float = 0.1,
        mut_prob: float = 0.05,
    ):
        self.low = low
        self.high = high
        self.mut_prob = mut_prob

    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        add_factor = torch.empty_like(chromosome).uniform_(self.low, self.high)
        mut_target = torch.rand_like(chromosome) <= self.mut_prob
        mutant = chromosome.clone()
        mutant[mut_target] += add_factor[mut_target]
        return mutant
        