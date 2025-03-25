from typing import List, Literal, Iterable
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale

from ._base import BaseSelection
from ._map_probability import ProbTypes, get_prob_mapper
from ._pareto_sort import AgglomerativeSelector


def torch_lexsort(a, dim=-1):
    assert dim == -1  # Transpose if you want differently
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
    # To be consistent with numpy, we flip the keys (sort by last row first)
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)

class MultiObjectiveSelection(BaseSelection):
    elite_num: int
    parents_num: int
    minimize: bool
    
    __multi_objective_checked: bool = False
    
    def best_indices(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def is_single_objective(self) -> bool:
        return False
    
    @property
    def is_multi_objective(self) -> bool:
        return True
    
    def _check_multi_objective(self, fitness: np.ndarray):
        """check if fitness is multi-objective (which is 2-dim array)"""
        if self.__multi_objective_checked:
            pass
        else:
            msg = 'Need 2-Dim fitness for Multi-Objective Selection Algorithm'
            assert fitness.ndim == 2, msg
            
            self.__multi_objective_checked = True
    

class LexsortSelection(MultiObjectiveSelection):
    """Lexicographical Sorting Selection. 
    fitness를 앞(index==0)부터 순서대로 우선순위를 매겨서 sorting 함. 
    """
    
    def __init__(
        self,
        elite_num: int,
        parents_num: int,
        prob_method: ProbTypes = 'softmax',
        minimize: bool = True,
    ):
        self.elite_num = elite_num
        self.parents_num = parents_num
        self.prob_method = prob_method
        self.minimize = minimize
        
        self.map_prob = get_prob_mapper(prob_method)
    
    
    def select(self, fitness: torch.Tensor):
        self._check_multi_objective(fitness)
        self.fitness = fitness

        # lexsort
        sort_order = reversed(range(self.fitness.shape[1]))  # 왜 lexsort는 키 순서를 거꾸로 받는가...
        sort_keys = torch.stack([self.fitness[:, i] for i in sort_order])

        if self.minimize:
            self.__sort_idx = torch_lexsort(sort_keys)
        else:
            self.__sort_idx = torch_lexsort(sort_keys)[::-1]
        
        # record elites' & parents' indices
        self.__elite_idx = self.__sort_idx[:self.elite_num]
        self.__parents_idx = self.__sort_idx[:self.parents_num]
        
        # calculate parents selection probability
        parents_fitness = self.fitness[self.__parents_idx]
        parents_rank = self._get_parents_rank(parents_fitness)
        self.__parents_prob = self.map_prob(parents_rank)
    
        # best_indices: best fitness indices
        # lexsort rank starts with 1
        self.__best_indices = self.__parents_idx[torch.where(parents_rank == 1)[0]]
    
    
    def _get_parents_rank(self, parents_fitness: torch.Tensor) -> torch.Tensor:
        cur_rank = 1
        rank_list = [cur_rank]
        last_fit = parents_fitness[0]
        for fit in parents_fitness[1:]:
            if not torch.equal(last_fit, fit):
                cur_rank += 1
                last_fit = fit
            rank_list.append(cur_rank)
        
        return torch.tensor(rank_list)
    
    
    def elite_idx(self) -> torch.Tensor:
        return self.__elite_idx
    
    
    # def pick_parents(self, n: int) -> torch.Tensor:
    #     assert n <= self.parents_num, \
    #         f"parents_num should be greater than or equal to {n}"
        
    #     indices = self.__parents_idx
    #     return np.random.choice(indices, size=n, replace=False, p=self.__parents_prob.numpy())
    
    def pick_parents(self, n: int, len_offspring: int) -> torch.Tensor:
        assert n <= self.parents_num, \
            f"parents_num should be greater than or equal to {n}"
        
        indices = self.__parents_idx
        return np.random.choice(indices, size=(len_offspring, n), replace=True, p=self.__parents_prob.numpy())


    def best_indices(self) -> torch.Tensor:
        return self.__best_indices

class ParetoSelection(MultiObjectiveSelection):
    """NSGA-II와 Agglomerative Clustering을 결합한 Pareto Sorting"""
    
    def __init__(
        self,
        elite_num: int,
        parents_num: int,
        prob_method: ProbTypes = 'softmax',
        minimize: bool=True,
    ):
        self.elite_num = elite_num
        self.parents_num = parents_num
        self.prob_method = prob_method
        self.minimize = minimize
        
        self.map_prob = get_prob_mapper(prob_method)
    
    
    def select(self, fitness: torch.Tensor):
        self._check_multi_objective(fitness)
        self.fitness = fitness
        
        # do pareto-sort (based on NSGA-II)
        selector = AgglomerativeSelector(
            self.fitness if self.minimize else -self.fitness,
            weight_on_extreme=True,
        )
        
        # record elites' & parents' indices
        self.__elite_idx, _ = selector.select(self.elite_num)
        self.__parents_idx, parents_scores = selector.select(self.parents_num)
        # parents_scores = torch.tensor(parents_scores)
        
        # calculate parents selection probability
        self.__parents_prob = self.map_prob(-parents_scores) # scores는 낮을수록 좋음
        
        # best_indices: best fitness indices based on scores
        scores = selector.scores
        self.__best_indices = torch.where(scores == scores.min())[0]
    
    def elite_idx(self) -> torch.Tensor:
        return self.__elite_idx
    
    # def pick_parents(self, n: int) -> torch.Tensor:
    #     assert n <= self.parents_num, \
    #         f"parents_num should be greater than or equal to {n}"
        
    #     indices = self.__parents_idx
    #     # return torch.multinomial(self.__parents_prob, n, replacement=False)
    #     return np.random.choice(indices, size=n, replace=False, p=self.__parents_prob.numpy())
    
    def pick_parents(self, n: int, len_offspring: int) -> torch.Tensor:
        assert n <= self.parents_num, \
            f"parents_num should be greater than or equal to {n}"
        
        indices = self.__parents_idx
        # return torch.multinomial(self.__parents_prob, n, replacement=False)
        return np.random.choice(indices, size=(len_offspring, n), replace=True, p=self.__parents_prob.numpy())
    
    def best_indices(self) -> torch.Tensor:
        return self.__best_indices
    

class ParetoLexsortSelection(MultiObjectiveSelection):
    """Pareto Selection과 Lexicographical Selection을 결합한 multi-objective selection"""
    
    def __init__(
        self,
        elite_num: int,
        parents_num: int,
        priority: List[int],
        prior_ratio: List[float] = [],
        prob_method: ProbTypes = 'softmax',
        minimize: bool = True,
    ):
        """Pareto-Lexicographical Selection
        
        Args:
            priority: List[int]
                fitness의 우선순위를 나타낸 list. 
                우선순위는 0부터 오름차순이며, 모든 fitness에 대해 표기되어야 함. 
                또한, 우선순위는 중복 가능.
            
            prior_ratio: List[float] (defaults to [])
                각 priority level에 대한 가중치. 
                empty list인 경우, priority 0번에 모든 가중치를 부여함.
            
            prob_method: ProbTypes (defaults to ProbTypes.SOFTMAX)
                parents 중에서 선발 시 확률 부여 로직

            minimize: bool (defaults to True)
                fitness 최적화 방향 설정. 
                minimize==True이면, 최소화 방향으로 selection 진행.
        """
    
        self.elite_num = elite_num
        self.parents_num = parents_num
        self.priority = priority
        self.prior_ratio = prior_ratio
        self.prob_method = prob_method
        self.minimize = minimize
        
        # map_prob: tensor를 probability tensor로 치환
        self.map_prob = get_prob_mapper(prob_method)
    
    def select(self, fitness: torch.Tensor):
        self._check_multi_objective(fitness)
        self.fitness = fitness
        
        # do pareto-sort (based on NSGA-II)
        selector = AgglomerativeSelector(
            self.fitness if self.minimize else -self.fitness,
            priority=self.priority,
            prior_ratio=self.prior_ratio,
            weight_on_extreme=True,
        )
        
        # record elites' & parents' indices
        self.__elite_idx, _ = selector.select(self.elite_num)
        self.__parents_idx, parents_scores = selector.select(self.parents_num)
        
        # calculate parents selection probability
        parents_scores = torch.tensor(minmax_scale(parents_scores.numpy()))
        self.__parents_prob = self.map_prob(-parents_scores) # scores는 낮을수록 좋음
        
        # best_indices: best fitness indices based on scores
        scores = selector.scores
        self.__best_indices = torch.where(scores == scores.min())[0]
    
    def elite_idx(self) -> torch.Tensor:
        return self.__elite_idx
    
    # def pick_parents(self, n: int) -> torch.Tensor:
    #     assert n <= self.parents_num, \
    #         f"parents_num should be greater than or equal to {n}"
        
    #     indices = self.__parents_idx
    #     return np.random.choice(indices, size=n, replace=False, p=self.__parents_prob.numpy())

    def pick_parents(self, n: int, len_offspring: int) -> torch.Tensor:
        assert n <= self.parents_num, \
            f"parents_num should be greater than or equal to {n}"
        
        indices = self.__parents_idx
        return np.random.choice(indices, size=(len_offspring, n), replace=True, p=self.__parents_prob.numpy())
    
    def best_indices(self) -> torch.Tensor:
        return self.__best_indices
    
    def sort_idx(self, fitness, sort_num):
        self._check_multi_objective(fitness)
        
        # do pareto-sort (based on NSGA-II)
        selector = AgglomerativeSelector(
            fitness if self.minimize else -fitness,
            priority=self.priority,
            prior_ratio=self.prior_ratio,
            weight_on_extreme=True,
        )
        elite_idx, _ = selector.select(sort_num)
        return elite_idx