from typing import Any, List, Callable, Tuple
from copy import deepcopy

import numpy as np
import torch

from ..callbacks import BaseCallback


class BaseMutation:
    
    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    # @staticmethod
    # def pick_by_rand(shape: Tuple[int], mut_prob: float, at_least: int = 1):
    #     """np.random.rand 기반 mut_target 만드는 함수
        
    #     Args:
    #         shape: chromosome shape
            
    #         mut_prob: threshold prob
            
    #         at_least: 최소 True 개수 (np.random.rand가 가장 낮은 순)
    #     """
    #     rand_arr = np.random.rand(*shape)
    #     mut_target = rand_arr <= mut_prob
        
    #     # at_least보다 mut_target==True의 개수가 적으면 부족한 분량 산출
    #     lack = at_least - sum(mut_target)
    #     # chromosome 전체 길이보다 at_least가 큰 경우 방지
    #     if lack > 0 and not np.prod(shape) <= at_least:
    #         rand_arr[mut_target] = 1.
    #         for _ in range(lack):
    #             target_idx = np.where(rand_arr==rand_arr.min())
    #             target_idx = tuple(map(lambda x: x[0], target_idx))
                
    #             mut_target[target_idx] = True
    #             rand_arr[target_idx] = 1.
        
    #     return mut_target

    @staticmethod
    def pick_by_rand(shape: Tuple[int, ...], mut_prob: float, at_least: int = 1):
        """np.random.rand 기반 mut_target 만드는 함수
        
        Args:
            shape: chromosome shape
            
            mut_prob: threshold prob
            
            at_least: 최소 True 개수 (np.random.rand가 가장 낮은 순)
        """
        rand_arr = np.random.rand(*shape)
        mut_target = rand_arr <= mut_prob
        
        # at_least보다 mut_target==True의 개수가 적으면 부족한 분량 산출
        lack = at_least - np.sum(mut_target)
        # chromosome 전체 길이보다 at_least가 큰 경우 방지
        if lack > 0 and not np.prod(shape, dtype=int) <= at_least:
            rand_arr[mut_target] = 1.
            for _ in range(lack):
                target_idx = np.unravel_index(np.argmin(rand_arr), shape)
                
                mut_target[target_idx] = True
                rand_arr[target_idx] = 1.
        
        return mut_target


class ChainMutation(BaseMutation):
    """여러 mutation 들을 순차적으로 진행할 때 사용하는 클래스"""
    
    def __init__(self, mutations: List[BaseMutation]):
        self.mutations = mutations
    
    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        # mutant = deepcopy(chromosome)
        mutant = chromosome
        
        for mut_ops in self.mutations:
            mutant = mut_ops(mutant)
        
        return mutant

    # code 보니 mutation에서는 train, step, being, end를 쓰지 않음 -> 제거
    # def on_train_begin(self, trainer) -> None:
    #     [mut.on_train_begin(trainer) for mut in self.mutations]
    
    # def on_train_end(self, trainer) -> None:
    #     [mut.on_train_end(trainer) for mut in self.mutations]
    
    # def on_step_begin(self, trainer) -> None:
    #     [mut.on_step_begin(trainer) for mut in self.mutations]
    
    # def on_step_end(self, trainer) -> None:
    #     [mut.on_step_end(trainer) for mut in self.mutations]


class FunctionMutation(BaseMutation):
    """Customizable Mutation by user-defined function
    
    Example:
        >>> def custom_mutation(chromosome):
        >>>     mutant = foo(chromosome)
        >>>     return mutant
        
        >>> mutation = FunctionMutation(custom_mutation)
    """
    
    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]):
        """Args:
            function: Callable
                mutation을 수행할 함수
        """
        self.function = function
        
    def __call__(self, chromosome: torch.Tensor) -> torch.Tensor:
        # return self.function(deepcopy(chromosome))
        return self.function(chromosome)
        
        