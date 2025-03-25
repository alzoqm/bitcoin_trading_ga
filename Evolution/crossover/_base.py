from typing import List, Callable
from copy import deepcopy

import numpy as np

from ..callbacks import BaseCallback


class BaseCrossover(BaseCallback):
    
    def __call__(self, parents: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def get_num_parents(self) -> int:
        # 특별히 overwrite되지 않으면 parents는 2개를 뽑음
        return 2


class SkipCrossover(BaseCrossover):
    """NO CROSSOVER; Crossover를 skip할 때 사용하는 클래스"""
    
    def __call__(self, parents: List[np.ndarray]) -> np.ndarray:
        # return parents[0]
        return parents[0]

    def get_num_parents(self) -> int:
        # mutation을 시행할 chromosome 1개만 선출함
        return 1


class FunctionCrossover(BaseCrossover):
    """Customizable Crossover by user-defined function
    
    Example:
        >>> def custom_crossover(parents_list):
        >>>     res_list = []
        >>>     for parent in parents_list:
        >>>         res_list.append(foo(parent))
        >>>     offspring = bar(res_list)
        >>>     return offspring
        
        >>> crossover = FunctionCrossover(custom_crossover, num_parents=10)
    """
    
    def __init__(self, function: Callable[[List[np.ndarray]], np.ndarray], num_parents: int = 2):
        """Args:
            function: Callable
                crossover를 수행할 함수
                
            num_parents: int
                crossover에 필요한 parents 수
        """
        self.function = function
        self.num_parents = num_parents
    
    def __call__(self, parents: List[np.ndarray]) -> np.ndarray:
        return self.function(parents)
    
    def get_num_parents(self) -> int:
        return self.num_parents