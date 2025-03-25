from typing import Literal

import torch
import numpy as np

ProbTypes = Literal['uniform', 'softmax', 'sparsemax']

def get_prob_mapper(prob_method: ProbTypes):
    if prob_method == 'uniform':
        return map_uniform
    elif prob_method == 'softmax':
        return map_softmax
    elif prob_method == 'sparsemax':
        return map_sparsemax
    else:
        raise NotImplementedError(f'prob_method {prob_method} is not Implemented')


# def map_uniform(arr: np.ndarray) -> np.ndarray:
#     """값에 상관없이 같은 확률 부여"""
#     prob = 1. / np.prod(arr.shape)
#     return np.full(arr.shape, fill_value=prob)


# def map_softmax(arr: np.ndarray) -> np.ndarray:
#     """softmax function
    
#     Note:
#         Bigger value, bigger prob
#     """
#     prob = np.exp(arr - arr.min())
#     prob = prob / prob.sum()
#     return prob


# def map_sparsemax(z: np.ndarray) -> np.ndarray:
#     """sparsemax function
    
#     Note:
#         Bigger value, bigger prob
    
#     References:
#         https://paperswithcode.com/method/sparsemax
#     """
#     # sort z
#     z_sorted = np.sort(z)[::-1]
    
#     # get k(z)
#     k = np.arange(z.shape[-1]) + 1
#     k_arr = 1 + k * z_sorted
    
#     z_cumsum = np.cumsum(z_sorted)
#     k_selected = k_arr > z_cumsum
    
#     k_max = np.where(k_selected)[0].max() + 1 # k(z)
    
#     # get t(z)
#     threshold = (z_cumsum[k_max - 1] - 1) / k_max # t(z)
    
#     # get sparsemax(z)
#     return np.maximum(z - threshold, 0)


def map_uniform(arr: torch.Tensor) -> torch.Tensor:
    """값에 상관없이 같은 확률 부여"""
    prob = 1. / torch.prod(torch.tensor(arr.shape).float())
    return torch.full_like(arr, fill_value=prob)


def map_softmax(arr: torch.Tensor) -> torch.Tensor:
    """softmax function
    
    Note:
        Bigger value, bigger prob
    """
    # prob = torch.exp(arr - arr.min())
    # prob = prob / prob.sum()
    # return prob
    return torch.softmax(arr.float(), dim=0)


def map_sparsemax(z: torch.Tensor) -> torch.Tensor:
    """sparsemax function
    
    Note:
        Bigger value, bigger prob
    
    References:
        https://paperswithcode.com/method/sparsemax
    """
    # sort z
    z_sorted, _ = torch.sort(z, descending=True)
    
    # get k(z)
    k = torch.arange(z.shape[-1], device=z.device) + 1
    k_arr = 1 + k * z_sorted
    
    z_cumsum = torch.cumsum(z_sorted, dim=-1)
    k_selected = k_arr > z_cumsum
    
    k_max = torch.where(k_selected)[0].max() + 1 # k(z)
    
    # get t(z)
    threshold = (z_cumsum[k_max - 1] - 1) / k_max # t(z)
    
    # get sparsemax(z)
    return torch.maximum(z - threshold, torch.tensor(0.0, device=z.device))
