from typing import Literal

import numpy as np
import torch

from ._base import BaseSelection
from ._map_probability import ProbTypes, get_prob_mapper



class SingleObjectiveSelection(BaseSelection):
    elite_num: int
    parents_num: int
    minimize: bool
    
    __single_objective_checked: bool = False
    
    def best_one(self) -> int:
        raise NotImplementedError
    
    @property
    def is_single_objective(self) -> bool:
        return True
    
    @property
    def is_multi_objective(self) -> bool:
        return False

    def _check_single_objective(self, fitness: np.ndarray):
        """check if fitness is single-objective (which is 1-dim array)"""
        if self.__single_objective_checked:
            pass
        else:
            msg = 'Need 1-Dim fitness for Single-Objective Selection Algorithm'
            assert fitness.ndim == 1, msg
            
            self.__single_objective_checked = True
    
    
class TournamentSelection(SingleObjectiveSelection):
    
    def __init__(
        self, 
        elite_num: int,
        parents_num: int,
        k: int = 6, 
        minimize: bool = True,
    ):
        self.elite_num = elite_num
        self.parents_num = parents_num
        self.k = k
        self.minimize = minimize
    
    def select(self, fitness: torch.Tensor):
        # check if fitness is single-objective
        self._check_single_objective(fitness)
        assert self.k <= fitness.size(0)
        self.fitness = fitness
        
        # sort by fitness
        if self.minimize:
            self.__sort_idx = torch.argsort(self.fitness)
        else:
            self.__sort_idx = torch.argsort(self.fitness, descending=True)
        
        # best_idx: best fitness index
        self.__best_idx = self.__sort_idx[0]
        
        # record elites' & parents' indices
        self.__elite_idx = self.__sort_idx[:self.elite_num]
        self.__parents_idx = self.__sort_idx[:self.parents_num]
    
    def elite_idx(self):
        return self.__elite_idx

    def pick_parents(self, n: int, len_offspring: int):
        assert (self.k + n - 1) <= self.parents_num, \
            f"parents_num should be greater than or equal to {self.k + n - 1}"
        
        # pick n parents
        indices = self.__parents_idx.repeat(len_offspring, 1)
    
        idx_list = []
        for _ in range(n):
            candidates = np.random.choice(len(indices[0]), replace=True, size=(len_offspring, self.k))
            if self.minimize:
                pick_idx = torch.argmin(self.fitness[candidates], dim=1)
            else:
                pick_idx = torch.argmax(self.fitness[candidates], dim=1)
            pick_idx = candidates[:, pick_idx][np.eye(len_offspring, len_offspring) == 1]
            pick_idx = indices[:, pick_idx][np.eye(len_offspring, len_offspring) == 1]
            idx_list.append(pick_idx)
            
        return torch.stack(idx_list).T

    def best_one(self) -> int:
        return self.__best_idx.item()


    
class RouletteSelection(SingleObjectiveSelection):
    
    def __init__(
        self, 
        elite_num: int,
        parents_num: int,
        prob_method: ProbTypes = 'uniform', 
        minimize: bool = True,
    ):
        self.elite_num = elite_num
        self.parents_num = parents_num
        self.prob_method = prob_method
        self.minimize = minimize
        self.map_prob = get_prob_mapper(prob_method)
    
    def select(self, fitness: torch.Tensor):
        # check if fitness is single-objective
        self._check_single_objective(fitness)
        self.fitness = fitness

        # sort by fitness
        if self.minimize:
            self.__sort_idx = torch.argsort(self.fitness).cpu()
        else:
            self.__sort_idx = torch.argsort(self.fitness, descending=True).cpu()
        
        # best_idx: best fitness index
        self.__best_idx = self.__sort_idx[0]
        
        # record elites' & parents' indices
        self.__elite_idx = self.__sort_idx[:self.elite_num]
        self.__parents_idx = self.__sort_idx[:self.parents_num]
        
        # calculate probability
        parents_fitness = self.fitness[self.__parents_idx]
        if self.minimize:
            self.__parents_prob = self.map_prob(-parents_fitness)
        else:
            self.__parents_prob = self.map_prob(parents_fitness)
        
    def elite_idx(self):
        return self.__elite_idx
    

    def sort_idx(self):
        return self.__sort_idx


    # def pick_parents(self, n: int) -> torch.Tensor:
    #     assert n <= self.parents_num, \
    #         f"parents_num should be greater than or equal to {n}"
        
    #     indices = self.__parents_idx
    #     # return torch.multinomial(self.__parents_prob, n, replacement=False)
    #     return np.random.choice(indices, size=n, replace=False, p=self.__parents_prob.numpy())

    def pick_parents(self, n: int, len_offspring: int):
        assert n <= self.parents_num, \
            f"parents_num should be greater than or equal to {n}"
        
        indices = self.__parents_idx
        # return torch.multinomial(self.__parents_prob, n, replacement=False)
        return np.random.choice(indices, size=(len_offspring, n), replace=True, p=self.__parents_prob.numpy() / sum(self.__parents_prob.numpy()))

    def best_one(self) -> int:
        return self.__best_idx.item()


class RandomSelection(RouletteSelection):
    """같은 확률로 parents selection 수행. 
    RouletteSelection(prob_method='uniform')와 동일함.
    """
    
    def __init__(
        self, 
        minimize: bool = True,
    ):
        self.prob_method = 'uniform'
        self.minimize = minimize
        self.map_prob = get_prob_mapper('uniform')