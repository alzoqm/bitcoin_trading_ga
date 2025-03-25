from typing import List
import numpy as np

from ..callbacks import BaseCallback


class BaseSelection(BaseCallback):
    elite_num: int
    parents_num: int
    minimize: bool
    
    def register_fitness(self, fitness: np.ndarray):
        """registers fitness"""
        raise NotImplementedError
    
    def elite_idx(self) -> np.ndarray:
        """returns indices of elite chromosomes"""
        raise NotImplementedError
    
    def pick_parents(self, n: int) -> np.ndarray:
        """returns n indices of parents
        
        Note:
            It does not return total indices of all parents
        """
        raise NotImplementedError
    
    def best_one(self) -> int:
        """returns if single-objective"""
        if self.is_single_objective:
            raise NotImplementedError
    
    def best_indices(self) -> np.ndarray:
        """returns if multi-objective"""
        if self.is_multi_objective:
            raise NotImplementedError

    @property
    def is_single_objective(self) -> bool:
        raise NotImplementedError
    
    @property
    def is_multi_objective(self) -> bool:
        raise NotImplementedError