from typing import Union, List, Callable
import warnings
import numpy as np
from ..callbacks._base import BaseCallback
from ..selection._pareto_sort import pareto_sort


class BaseEarlyStopping(BaseCallback):
    disabled: bool = False
    
    def on_train_begin(self, trainer) -> None:
        
        # if trainer가 val_fitness로 best_fit을 저장하는 경우
        if trainer.do_validate and not trainer.deterministic_validation:
            warnings.warn(
                'Validation is not deterministic, EarlyStopping disabled',
                UserWarning,
            )
            self.disabled = True
            
        # elif trainer가 fitness로 best_fit을 저장하는 경우
        elif not trainer.do_validate and not trainer.deterministic_evaluation:
            warnings.warn(
                'Evaluation is not deterministic, EarlyStopping disabled',
                UserWarning,
            )
            self.disabled = True


class SingleEarlyStopping(BaseEarlyStopping):
    """EarlyStopping for single-objective task
    
    Args:
        patience: int (defaults to -1) 
            early stopping의 기준이 되는 
            개선이 안 된 횟수의 threshold입니다. 
            만약 음수라면 사용되지 않습니다. 
        
        baseline: int | float | None (defaults to None) 
            fitness의 목표치 입니다. 
            best fitness가 목표치에 도달하면 종료합니다. 
            만약 None이라면 사용되지 않습니다. 
        
    Note:
        selection의 minimize에 따라 동작이 달라집니다. 
        minimize인 경우 np.less로 개선여부를 판단하고, 
        maximize인 경우 np.greater로 개선여부를 판단합니다. 
    """
    
    def __init__(
        self, 
        patience: int = -1, 
        baseline: Union[int, float, None] = None,
    ):
        self.patience = patience
        self.baseline = baseline
        
        self.wait: int = 0
        self.compare_op: Callable = None
        self.best_fit: Union[int, float] = None
        
        
    def on_train_begin(self, trainer) -> None:
        super().on_train_begin(trainer)
        
        # minimize, maximize에 따라 다름
        minimize = trainer.selection.minimize
        if minimize:
            self.compare_op = np.less
            self.best_fit = np.inf
        else:
            self.compare_op = np.greater
            self.best_fit = -np.inf
        

    def on_step_end(self, trainer) -> None:
        old_fit = self.best_fit
        new_fit = trainer.best_fitness
        
        # compare fitness
        if self.compare_op(new_fit, old_fit):
            self.best_fit = new_fit
            self.wait = 0
            
            # compare if fitness reached baseline
            if self.baseline and self.compare_op(new_fit, self.baseline):
                self.wait = self.patience
            
        else:
            self.wait += 1
        
        # stop train
        if self.disabled:
            # 만약 disabled 되었다면 pass
            pass
        elif self.wait == self.patience:
            # wait가 patience와 값이 같다면 stop_train = True
            trainer.stop_train = True


class MultiEarlyStopping(BaseCallback):
    """아이디어 제공 환영합니다!
    
    현재 계획은 각 fitness 별 평균값/frontier 값들을 저장해두고, 
    best_fitness들의 각 값들에 개선이 있었는지 여부를 보고 
    만약 하나라도 개선이 있었다면 improved, 
    하나도 개선이 없었다면 unimproved로 할까 고민 중입니다.
    
    여기서 문제점은, multi-objective이기 때문에 
    각 fitness들이 서로 trade-off 관계에 있을 수 있다는 점으로, 
    이로 인해 한 fitness의 평균값/frontier가 개선되더라도 
    다른 fitness들의 평균값/frontier가 악화되는 현상이 발생 가능합니다. 
    
    따라서 이 모듈이 담당하는 범위나 여타 내용들을 한정하고, 
    기능 개발에 들어갈 듯 합니다.
    """
    
    def __init__(self):
        raise NotImplementedError



class LexsortEarlyStopping(BaseCallback):
    """Lexicographical EarlyStopping for multi-objective task
    
    Args:
        patience: int
            Number of steps with no improvement 
            after which training will be stopped.
    
    Note:
        selection의 minimize에 따라 동작이 달라집니다.
        minimize인 경우 np.less로 개선여부를 판단하고, 
        maximize인 경우 np.greater로 개선여부를 판단합니다. 
        
    Note:
        np.lexsort와 비슷한 로직으로, 
        fitness의 앞부터 비교하여 개선여부를 판단합니다.
    """
    
    def __init__(self, patience: int):
        self.patience = patience

        self.wait = 0
        self.compare_op: Callable = None
        self.best_fit: np.ndarray = None
    
    
    def on_train_begin(self, trainer) -> None:
        super().on_train_begin(trainer)

        # minimize, maximize에 따라 다름
        minimize = trainer.selection.minimize
        if minimize:
            self.compare_op = np.less
            self.best_fit = np.full((1000,), np.inf)
        else:
            self.compare_op = np.greater
            self.best_fit = np.full((1000,), -np.inf)
    
    
    def on_step_end(self, trainer) -> None:
        old_fit = self.best_fit
        new_fit = trainer.best_fitness
        new_fit = self._modify_fitness(new_fit)
        
        # compare fitness
        if self._is_lexsort_improved(new_fit, old_fit):
            self.best_fit = new_fit
            self.wait = 0
        else:
            self.wait += 1
        
        # stop train
        if self.disabled:
            # 만약 disabled 되었다면 pass
            pass
        elif self.wait == self.patience:
            # wait가 patience와 값이 같다면 stop_train = True
            trainer.stop_train = True
    
    
    def _modify_fitness(self, fitness: np.ndarray):
        assert isinstance(fitness, np.ndarray), \
            '[LexsortEarlyStopping] fitness must be numpy array'
        
        if fitness.ndim == 2:
            return fitness[0]
        
        elif fitness.ndim == 1:
            return fitness
        
        else:
            raise AssertionError("[LexsortEarlyStopping] ndim of fitness must be less than 3")

    
    def _is_lexsort_improved(
        self, 
        new_fit_arr: np.ndarray, 
        old_fit_arr: np.ndarray,
    ) -> bool:
        # fitness의 앞에서부터 순서대로 비교
        for i in range(len(new_fit_arr)): # new_fit 기준으로 indexing
            new_fit, old_fit = new_fit_arr[i], old_fit_arr[i]
            if new_fit == old_fit:
                # i번째 fitness가 같다면 continue
                continue
            
            if self.compare_op(new_fit_arr[i], old_fit_arr[i]):
                # new_fit_arr의 i번째 fitness가 더 좋다면 return True
                return True
            else:
                return False
        
        # 모든 fitness가 같다면 False (개선되지 않았으므로)
        return False


class ParetoEarlyStopping(BaseEarlyStopping):
    """Pareto Based EarlyStopping for multi-objective task
    
    현재까지의 Best Fitness들의 Pareto Front를 기준으로, 
    새로운 Best Fitness가 기존의 Pareto Front의 원소들 중 
    dominate하는 경우가 있는지 여부로 개선 여부를 판단함. 
    
    Args:
        patience: int
            Number of steps with no improvement 
            after which training will be stopped.
    
    Note:
        아이디어 제공 환영합니다!
        
        원래 계획은 두 가지 관점에서 개선이 되었는지를 보려 했습니다. 
        1. 기존의 pareto front가 더 진보했는가? 
        2. 기존의 pareto front가 더 확장되었는가? 
        
        1번의 경우, EarlyStopping에서 기록하고 있는 pareto front의 fitness들 중 
        새로 들어온 fitness에 의해 pareto front에서 탈락되는 fitness가 있는지 여부가 
        기준이 됩니다. 
        
        2번의 경우, pareto front의 각 fitness들의 최적값들이 
        기존에 비해 더 확장되었는지 여부가 기준이 됩니다. 
    """
    
    def __init__(self, patience: int):
        self.patience = patience

        self.wait = 0
        self.minimize: bool = True
        self.pareto_front: np.ndarray = None
    
    
    def on_train_begin(self, trainer) -> None:
        super().on_train_begin(trainer)
        self.minimize = trainer.selection.minimize
    
    
    def on_step_end(self, trainer) -> None:
        new_best_fit = trainer.best_fitness
        is_updated = self._update_front(new_best_fit)
        
        # count wait
        if is_updated:
            self.wait = 0
        else:
            self.wait += 1
        
        # stop train
        if self.disabled:
            # 만약 disabled 되었다면 pass
            pass
        elif self.wait == self.patience:
            # wait가 patience와 값이 같다면 stop_train = True
            trainer.stop_train = True
    
    
    def _update_front(self, new_best_fit: np.ndarray) -> bool:
        
        old_front = self.pareto_front
        
        # 예외처리: 기존에 front가 없는 경우
        if old_front is None:
            self.pareto_front = new_best_fit.copy()
            is_updated = True
            return is_updated
        
        fit_concat = np.concatenate((old_front, new_best_fit))
        if self.minimize:
            rank_concat = pareto_sort(fit_concat)
        else:
            rank_concat = pareto_sort(-fit_concat)
        
        # 기존 old_front의 원소가 하나라도 탈락되면 is_updated = True
        is_updated = np.any(rank_concat[:len(old_front)] != 1)
        
        # update front
        new_front = fit_concat[rank_concat==1]
        new_front = np.unique(new_front, axis=0)
        self.pareto_front = new_front

        return is_updated
