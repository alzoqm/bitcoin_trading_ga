__all__ = [
    'pareto_sort',
    'lexico_sort',
    'AgglomerativeSelector',
]

from typing import List, Tuple

import numpy as np
from numba import njit
import torch

from sklearn.preprocessing import minmax_scale
from sklearn.cluster import AgglomerativeClustering


@torch.jit.script
def all_axis0(tensor: torch.Tensor):
    return torch.all(tensor, dim=1)

@torch.jit.script
def any_axis0(tensor: torch.Tensor):
    return torch.any(tensor, dim=1)

@torch.jit.script
def pareto_sort(fitness: torch.Tensor) -> torch.Tensor:
    '''
    NSGA-II's fast non-dominated sorting algorithm

    fitness is a 2d-tensor of individual fitness values,
    assuming smaller values are better

    Returns the rank for each row in fitness
    '''
    len_fitness = fitness.shape[0]
    rank_tensor = torch.zeros(len_fitness, dtype=torch.long)
    S = torch.zeros((len_fitness, len_fitness), dtype=torch.bool)
    n = torch.zeros(len_fitness, dtype=torch.long)

    for p in range(len_fitness):
        dominates_all = all_axis0(fitness[p] <= fitness)
        dominates_any = any_axis0(fitness[p] < fitness)

        # finding where p dominates
        S[p] = torch.logical_and(dominates_all, dominates_any)

        # finding where p is dominated
        n[p] = torch.logical_and(~dominates_all, ~dominates_any).sum()
        if n[p] == 0:
            rank_tensor[p] = 1
    
    rank = 1
    while torch.any(n > 0):
        front_sum = S[torch.where(rank_tensor == rank)[0]].sum(0)
        n -= front_sum

        rank += 1
        rank_tensor[torch.logical_and(n == 0, front_sum > 0)] = rank

    return rank_tensor


def lexico_sort(fitness: torch.Tensor) -> torch.Tensor:
    """Lexicographical fitness sorting algorithm
    
    fitness는 개별 fitness 값들을 2d-tensor로 묶은 형태이고, 
    값이 작을 수록 좋다고 가정함.
    또한 왼쪽 열부터 우선순위가 높다고 가정함. 
    
    fitness의 각 행에 대한 rank를 반환함
    """
    # lexsort ranking
    rank = torch.zeros(fitness.size(0), dtype=torch.int64, device=fitness.device)
    
    # PyTorch equivalent of np.unique
    unique_sorted, inverse_indices = torch.unique(fitness, dim=0, sorted=True, return_inverse=True)
    
    for i in range(unique_sorted.size(0)):
        # PyTorch equivalent of np.all(fitness == unique_sorted, axis=1)
        mask = (inverse_indices == i)
        rank[mask] = i + 1

    return rank

class AgglomerativeSelector:
    """Agglomerative Clustering 기반 pareto-lexicographical selection"""
    
    def __init__(
        self, 
        fitness: np.ndarray, 
        priority: List[int] = [],
        prior_ratio: List[float] = [],
        weight_on_extreme: bool = False,
    ):
        """Agglomerative Pareto-Lexicographical Selection
        
        Args:
            fitness: np.ndarray
                fitness. 작을 수록 최적화 가정.
            
            priority: List[int]
                각 fitness 별 우선순위 (낮을수록 우선)
            
            prior_ratio: List[float]
                각 우선순위 별 비율. 0 ~ 1 사이.
            
            weight_on_extreme: bool
                개별 최적 fitness에 대해 가중치를 줄 것인지 여부.
        """
        self.fitness = torch.tensor(minmax_scale(fitness.numpy()))
        # 개별 최적 fitness에 가중치 부여
        if weight_on_extreme:
            weight = self.fitness.shape[-1] # fitness의 개수만큼 weight 부여
            # 개별 최적 fitness에 weight만큼 뺌
            for i in range(self.fitness.shape[-1]):
                argmin_mask = self.fitness[:, i] == self.fitness[:, i].min()
                self.fitness[argmin_mask, i] -= weight
        
        self.priority = self._check_priority(self.fitness, priority)
        self.prior_ratio = self._check_prior_ratio(self.priority, prior_ratio)
        
        # sub_rank: 각 priority 별 pareto rank
        self.sub_rank = self._get_sub_rank(self.priority, self.fitness)
        # scores: sub_rank를 고려한 각 fitness 별 scores
        self.scores = self._get_scores(self.sub_rank)
    

    def select(self, num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """num 만큼 select하여 index와 score를 반환하는 메소드
        
        각 priority 마다의 pareto rank를 R0, R1, R2, ... 라고 하고, 
        각 prior_ratio를 r0, r1, r2, ... 라고 하자. 
        이 때, select해야하는 개수(num)와 prior_ratio의 cumsum의 곱을 계산하여, 
        이를 n0, n1, n2, ... 라고 하자.
        
        R0, R1, R2, ... 전부를 고려하여 n0만큼 lexsort selection을 진행한 뒤, 
        R0과 R1에 대하여 pareto sorting을 시행하여 pareto rank R01를 만든다. 
        그 후, R01, R2, ... 를 고려하여 n1-n0만큼 lexsort selection을 진행한 뒤, 
        R01과 R2에 대하여 pareto rank R012를 만든다.
        
        이를 반복하여 selection을 진행한다.
        
        Args:
            num: int
                selection number
        
        Returns:
            (selected indices, score): Tuple[Tensor, Tensor]
                선출된 인덱스와 각 인덱스 별 스코어 (스코어는 낮을 수록 좋음)
        """
        
        # 예외처리: num >= len(fitness)
        if num >= self.fitness.shape[0]:
            sort_idx = torch.argsort(self.scores)
            return sort_idx, self.scores[sort_idx].clone()
        
        # convert ratio to cumulative numbers
        prior_num = (torch.cumsum(self.prior_ratio, dim=0) * num).long()
        prior_num[-1] = num
        
        # select
        sub_rank = self.sub_rank.clone()
        select_idx = []
        # select_num: select_idx의 목표 길이
        for select_num in prior_num:
            
            # 이미 뽑힌 인덱스의 rank는 0으로 변경
            lexsort_rank = lexico_sort(sub_rank)
            if select_idx:
                lexsort_rank[torch.tensor(select_idx)] = 0
            
            current_rank = 1
            num_lack = select_num - len(select_idx) # num_lack: select_idx에 더 채워야 하는 개수
            for current_rank in torch.unique(lexsort_rank):
                if current_rank <= 0: continue
                elif num_lack == 0: break
                
                # select_target: indices where lexsort_rank is current_rank
                select_target = torch.where(lexsort_rank==current_rank)[0]
                
                # 만약 select_target이 num_lack 보다 작거나 같으면 그대로 select_idx에 추가
                if len(select_target) <= num_lack:
                    select_idx += select_target.tolist()
                
                # 아니라면, select_target의 fitness를 clustering하여 random select
                else:
                    clustering = AgglomerativeClustering(n_clusters=int(num_lack))
                    target_fitness = self.fitness[select_target].cpu().numpy()
                    
                    cluster_labels = clustering.fit_predict(target_fitness)
                    # draw index
                    for label in range(num_lack):
                        label_indices = torch.where(torch.tensor(cluster_labels)==label)[0]
                        # idx = torch.randint(len(label_indices), (1,)).item()
                        idx = label_indices[0].item() # 랜덤성 제거
                        select_idx.append(select_target[idx].item())
                
                # update current_rank
                current_rank += 1
                # update num_lack
                num_lack = select_num - len(select_idx)
            
            # do not update if last iteration
            if sub_rank.shape[-1] >= 2:
                # update sub_rank
                sub_rank = torch.column_stack((pareto_sort(sub_rank[:, :2]), sub_rank[:, 2:]))
        
        select_idx = torch.tensor(select_idx)
        return select_idx, self.scores[select_idx].clone()
    
    
    def _check_priority(self, fitness: torch.Tensor, priority=[]) -> torch.Tensor:
        # priority가 fitness에 부합하는지 여부
        if priority:
            # check shape
            assert len(priority) == fitness.shape[-1], \
                "length of priority != number of fitness"
            # priority 숫자 보정 ([0, 1, 3] -> [0, 1, 2])
            unique_priority = torch.unique(torch.tensor(priority))
            mapper = {x.item(): i for i, x in enumerate(unique_priority)}
            priority = [mapper[p] for p in priority]
        else:
            priority = [0 for _ in range(fitness.shape[-1])]
        
        # return as PyTorch tensor
        return torch.tensor(priority, device=fitness.device)
    
    def _check_prior_ratio(self, priority: torch.Tensor, prior_ratio=[]) -> torch.Tensor:
        if prior_ratio:
            # check length
            needed_length = priority.max().item() + 1
            if len(prior_ratio) < needed_length:
                prior_ratio += [0. for _ in range(needed_length - len(prior_ratio))]
            elif len(prior_ratio) > needed_length:
                prior_ratio = prior_ratio[:needed_length]
            
            # clip ratio into 0 to 1
            prior_ratio = torch.tensor(prior_ratio, device=priority.device).clamp(0., 1.)
            # normalize ratio
            prior_ratio /= prior_ratio.sum()
        else:
            # default prior_ratio: [1., 0., 0., ...]
            prior_ratio = torch.zeros(priority.max().item() + 1, dtype=torch.float, device=priority.device)
            prior_ratio[0] = 1.
        
        # return as PyTorch tensor
        return prior_ratio
    
    def _get_priority_idx_list(self, priority: torch.Tensor) -> List[torch.Tensor]:
        """
        priority index list
        priority가 [0, 0, 1, 1, 2] 일 때,
        priority index list는 [[0, 1], [2, 3], [4]]
        """
        idx_list = []
        for p in torch.unique(priority):
            p_idx = torch.where(priority == p)[0]
            idx_list.append(p_idx)
        return idx_list
    
    def _get_sub_rank(self, priority: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        # 각 priority 별 rank를 산출
        rank_list = []
        for priority_idx in self._get_priority_idx_list(priority):
            rank = pareto_sort(fitness[:, priority_idx])  # Assuming pareto_sort is implemented for PyTorch
            rank_list.append(rank)
        
        return torch.stack(rank_list, dim=1)
    
    def _get_scores(self, sub_rank: torch.Tensor) -> torch.Tensor:
        # sub_rank에서 score 산출
        # 첫번째 priority rank를 정수단위로, 그 아래를 소수단위로 계산함
        scores = sub_rank[:, 0].float()
        
        degree = 0
        for i in range(1, sub_rank.shape[-1]):
            degree -= len(str(sub_rank[:, i].max().item()))
            scores += sub_rank[:, i].float() * (10**degree)
        
        return scores