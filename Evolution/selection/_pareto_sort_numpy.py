__all__ = [
    'pareto_sort',
    'lexico_sort',
    'AgglomerativeSelector',
]

from typing import List, Tuple

import numpy as np
from numba import njit

from sklearn.preprocessing import minmax_scale
from sklearn.cluster import AgglomerativeClustering


@njit(cache=True)
def all_axis0(array):
    # assume array as 2-dim ndarray
    bool_arr = np.zeros(array.shape[0], dtype=np.bool8)
    for i in range(array.shape[0]):
        bool_arr[i] = np.all(array[i])
    return bool_arr

@njit(cache=True)
def any_axis0(array):
    # assume array as 2-dim ndarray
    bool_arr = np.zeros(array.shape[0], dtype=np.bool8)
    for i in range(array.shape[0]):
        bool_arr[i] = np.any(array[i])
    return bool_arr

@njit(cache=True)
def pareto_sort(fitness: np.ndarray) -> np.ndarray:
    '''
    NSGA-II의 fast non-dominated sorting algorithm

    fitness는 개별 fitness 값들을 2d-array로 묶은 형태이고, 
    값이 작을 수록 좋다고 가정함

    fitness의 각 행에 대한 rank를 반환함
    '''
    len_fitness = fitness.shape[0]
    rank_arr = np.zeros(len_fitness, dtype=np.intp)
    S = np.zeros((len_fitness, len_fitness), np.bool8)
    n = np.zeros(len_fitness, dtype=np.intp)

    for p in range(len_fitness):
        
        dominates_all = all_axis0(fitness[p] <= fitness)
        dominates_any = any_axis0(fitness[p] < fitness)

        # finding where p dominates
        S[p] = np.logical_and(dominates_all, dominates_any)

        # finding where p is dominated
        n[p] = np.logical_and(~dominates_all, ~dominates_any).sum()
        if n[p] == 0: rank_arr[p] = 1
            
    # n_value = n.copy()
    rank = 1
    while np.any(n > 0):
        front_sum = S[np.where(rank_arr == rank)[0]].sum(0)
        n -= front_sum

        rank += 1
        rank_arr[np.logical_and(n == 0, front_sum > 0)] = rank

    return rank_arr


def lexico_sort(fitness: np.ndarray) -> np.ndarray:
    """Lexicographical fitness sorting algorithm
    
    fitness는 개별 fitness 값들을 2d-array로 묶은 형태이고, 
    값이 작을 수록 좋다고 가정함.
    또한 왼쪽 열부터 우선순위가 높다고 가정함. 
    
    fitness의 각 행에 대한 rank를 반환함
    """
    # lexsort ranking
    rank = np.zeros(len(fitness), dtype=int)
    
    # np.unique는 sorting된 결과를 반환함
    for i, unique_sorted in enumerate(np.unique(fitness, axis=0)):
        # unique_sorted와 같은 fitness에 rank (i+1) 부여
        rank[np.all(fitness == unique_sorted, axis=1)] = i+1

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
        self.fitness = minmax_scale(fitness)
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
    

    def select(self, num: int) -> Tuple[np.ndarray, np.ndarray]:
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
            (selected indices, score): Tuple[ndarray, ndarray]
                선출된 인덱스와 각 인덱스 별 스코어 (스코어는 낮을 수록 좋음)
        """
        
        # 예외처리: num >= len(fitness)
        if num >= len(self.fitness):
            sort_idx = np.argsort(self.scores)
            return sort_idx, self.scores[sort_idx].copy()
        
        # convert ratio to cumulative numbers
        prior_num = (self.prior_ratio.cumsum() * num).astype(int)
        prior_num[-1] = num
        
        # select
        sub_rank = self.sub_rank.copy()
        select_idx = []
        # select_num: select_idx의 목표 길이
        for select_num in prior_num:
            
            # 이미 뽑힌 인덱스의 rank는 0으로 변경
            lexsort_rank = lexico_sort(sub_rank)
            lexsort_rank[select_idx] = 0
            
            current_rank = 1
            num_lack = select_num - len(select_idx) # num_lack: select_idx에 더 채워야 하는 개수
            for current_rank in np.unique(lexsort_rank):
                if current_rank <= 0: continue
                elif num_lack == 0: break
                
                # select_target: indices where lexsort_rank is current_rank
                select_target = np.where(lexsort_rank==current_rank)[0]
                
                # 만약 select_target이 num_lack 보다 작거나 같으면 그대로 select_idx에 추가
                if len(select_target) <= num_lack:
                    select_idx += list(select_target)
                
                # 아니라면, select_target의 fitness를 clustering하여 random select
                else:
                    clustering = AgglomerativeClustering(n_clusters=num_lack)
                    target_fitness = self.fitness[select_target]
                    print(target_fitness)
                    cluster_labels = clustering.fit_predict(target_fitness)
                    # draw index
                    for label in range(num_lack):
                        label_indices = np.where(cluster_labels==label)[0]
                        # idx = np.random.choice(label_indices)
                        idx = label_indices[0] # 랜덤성 제거
                        select_idx.append(select_target[idx])
                
                # update current_rank
                current_rank += 1
                # update num_lack
                num_lack = select_num - len(select_idx)
            
            # do not update if last iteration
            if sub_rank.shape[-1] >= 2:
                # update sub_rank
                sub_rank = np.column_stack((pareto_sort(sub_rank[:, :2]), sub_rank[:, 2:]))
        
        select_idx = np.array(select_idx)
        return select_idx, self.scores[select_idx].copy()
    
    
    def _check_priority(self, fitness, priority=[]) -> np.ndarray:
        # priority가 fitness에 부합하는지 여부
        if priority:
            # check shape
            assert len(priority) == fitness.shape[-1], \
                "length of priority != number of fitness"
            # priority 숫자 보정 ([0, 1, 3] -> [0, 1, 2])
            mapper = dict([(x, i) for i, x in enumerate(np.unique(priority))])
            priority = list(map(mapper.__getitem__, priority))
        else:
            priority = [0 for _ in range(fitness.shape[-1])]
        
        # return as numpy array
        return np.array(priority)
    
    
    def _check_prior_ratio(self, priority: np.ndarray, prior_ratio=[]) -> np.ndarray:
        if prior_ratio:
            # check length
            needed_length = max(priority) + 1
            if len(prior_ratio) < needed_length:
                prior_ratio += [0. for _ in range(needed_length - len(prior_ratio))]
            elif len(prior_ratio) > needed_length:
                prior_ratio = prior_ratio[:needed_length]
            
            # clip ratio into 0 to 1
            prior_ratio = np.clip(prior_ratio, 0., 1.)
            # normalize ratio
            prior_ratio /= prior_ratio.sum()
        else:
            # default prior_ratio: [1., 0., 0., ...]
            prior_ratio = np.zeros(max(priority)+1, dtype=float)
            prior_ratio[0] = 1.
        
        # return as numpy array
        return prior_ratio
    
    
    def _get_priority_idx_list(self, priority: np.ndarray) -> List[np.ndarray]:
        """priority index list
        priority가 [0, 0, 1, 1, 2] 일 때,
        priority index list는 [[0, 1], [2, 3], [4]]
        """
        idx_list = []
        for p in np.unique(priority):
            p_idx = np.where(priority == p)[0]
            idx_list.append(p_idx)
        return idx_list
    
    def _get_sub_rank(self, priority: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        # 각 priority 별 rank를 산출
        rank_list = []
        for priority_idx in self._get_priority_idx_list(priority):
            rank = pareto_sort(fitness[:, priority_idx])
            rank_list.append(rank)
        
        return np.stack(rank_list, axis=1)
    
    def _get_scores(self, sub_rank: np.ndarray) -> np.ndarray:
        # sub_rank에서 score 산출
        # 첫번째 priority rank를 정수단위로, 그 아래를 소수단위로 계산함
        scores = sub_rank[:, 0].astype(float)
        
        degree = 0
        for i in range(1, sub_rank.shape[-1]):
            degree -= len(str(sub_rank[:, i].max()))
            scores += sub_rank[:, i] * (10**degree)
        
        return scores