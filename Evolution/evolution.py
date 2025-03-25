from typing import Any, Callable, List, Union
from numbers import Integral, Real
from copy import deepcopy

import torch
import numpy as np

from tqdm import tqdm

class Evolution:
    def __init__(self, prescriptor, selection, crossover, mutation, group_size):
        self.prescriptor = prescriptor
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        
        self.chromosome_size = group_size
        self.num_parents = self.crossover.get_num_parents()

        self.check_model_shape()

    # def check_model_shape(self):
    #     device = next(self.prescriptor.parameters()).device
    #     self.prescriptor.cpu()
    #     self.shape_each_layer = []
    #     self.num_each_layer = []
    #     for name, param in self.prescriptor.layers[0].named_parameters():
    #         size = list(param.size())
    #         self.shape_each_layer.append(size)
    #         layer_param = 1
    #         for idx, item in enumerate(size):
    #             layer_param *= item
    #         self.num_each_layer.append(layer_param)

    #     self.prescriptor = self.prescriptor.to(device)
    
    def check_model_shape(self):
        device = next(self.prescriptor.parameters()).device
        self.prescriptor.cpu()
        # Base layers
        self.shape_each_layer = []
        self.num_each_layer = []
        for name, param in self.prescriptor.base_layers.named_parameters():
            size = list(param.size())
            self.shape_each_layer.append(size)
            self.num_each_layer.append(param.numel())

        # After layers
        self.after_shape_each_layer = []
        self.after_num_each_layer = []
        for name, param in self.prescriptor.after_layers.named_parameters():
            size = list(param.size())
            self.after_shape_each_layer.append(size)
            self.after_num_each_layer.append(param.numel())

        self.prescriptor = self.prescriptor.to(device)
    
    def update_chromosomes(self, chromosomes, base_shape, after_shape, device='cpu'):
        chromosomes_size = len(chromosomes)
        base_chromosomes = chromosomes[:, :base_shape[1]]
        after_chromosomes = chromosomes[:, base_shape[1]:]
        with torch.no_grad():
            # Update base layers
            
            sd = self.prescriptor.base_layers.state_dict()
            split_base = 0
            for idx_sd, param_name in enumerate(sd):
                split_margin = split_base + self.num_each_layer[idx_sd] // chromosomes_size
                param = base_chromosomes[:, split_base:split_margin].reshape(self.shape_each_layer[idx_sd])
                sd[param_name] = param
                split_base = split_margin
            self.prescriptor.base_layers.load_state_dict(sd)

            sd = self.prescriptor.after_layers.state_dict()
            split_base = 0
            for idx_sd, param_name in enumerate(sd):
                split_margin = split_base + self.after_num_each_layer[idx_sd] // chromosomes_size
                param = after_chromosomes[:, split_base:split_margin].reshape(self.after_shape_each_layer[idx_sd])
                sd[param_name] = param
                split_base = split_margin
            self.prescriptor.after_layers.load_state_dict(sd)
        self.prescriptor.to(device)


    def flatten_chromosomes(self):
        base_chromosomes, device = self.base_flatten_chromosomes()
        after_chromosomes, _ = self.after_flatten_chromosomes()

        base_ch_shape = base_chromosomes.shape
        after_ch_shape = after_chromosomes.shape
        chromosomes = torch.cat([base_chromosomes, after_chromosomes], dim=1)
        return chromosomes.cpu(), base_ch_shape, after_ch_shape, device

    def base_flatten_chromosomes(self,):
        device = next(self.prescriptor.parameters()).device
        chromosomes_size = self.prescriptor.num_blcoks
        self.prescriptor.cpu()
        with torch.no_grad():
            chromosomes = []
            ch = self.prescriptor.base_layers
            for name, param in ch.named_parameters():
                param = param.flatten().reshape(chromosomes_size, -1)
                chromosomes.append(param)
            chromosomes = torch.concat(chromosomes, dim=1)
        return chromosomes, device

    
    def after_flatten_chromosomes(self, ):
        device = next(self.prescriptor.parameters()).device
        chromosomes_size = self.prescriptor.num_blcoks
        self.prescriptor.cpu()
        with torch.no_grad():
            chromosomes = []
            ch = self.prescriptor.after_layers
            for name, param in ch.named_parameters():
                param = param.flatten().reshape(chromosomes_size, -1)
                chromosomes.append(param)
            chromosomes = torch.concat(chromosomes, dim=1)
        return chromosomes, device
    
    def verify_base_parameters(self):
        original_params = []
        for param in self.prescriptor.base_layers.parameters():
            original_params.append(param.clone())

        # Flatten and reload chromosomes
        chromosomes, base_ch_shape, after_ch_shape, device = self.flatten_chromosomes()
        self.update_chromosomes(chromosomes, base_ch_shape, after_ch_shape, device)

        # Check if parameters are the same
        for original_param, param in zip(original_params, self.prescriptor.base_layers.parameters()):
            if not torch.allclose(original_param, param):
                print("Parameters do not match base reload.")
                return False
        print("Parameters match base reload.")
        return True

    def verify_after_parameters(self):
        original_params = []
        for param in self.prescriptor.after_layers.parameters():
            original_params.append(param.clone())

        # Flatten and reload chromosomes
        chromosomes, base_ch_shape, after_ch_shape, device = self.flatten_chromosomes()
        self.update_chromosomes(chromosomes, base_ch_shape, after_ch_shape, device)

        # Check if parameters are the same
        for original_param, param in zip(original_params, self.prescriptor.after_layers.parameters()):
            if not torch.allclose(original_param, param):
                print("Parameters do not match after reload.")
                return False
        print("Parameters match after reload.")
        return True


    
    def select_elite(self, fitness: torch.Tensor, chromosomes: torch.Tensor, num_elite_chromosomes: int):
        self.selection.select(fitness)
        elite_idx = self.selection.sort_idx()[:num_elite_chromosomes] # for single
        # elite_idx = self.selection.sort_idx(fitness, num_elite_chromosomes).long() # for multi
        elite_chromosomes = chromosomes[elite_idx]
                
        return elite_idx, elite_chromosomes


    def evolve(self, fitness: torch.Tensor):
        # chromosomes = self.prescriptor.chromosomes.cpu()
        chromosomes, base_ch_shape, after_ch_shape, device = self.flatten_chromosomes()
        
        self.selection.select(fitness)
        elite_idx = self.selection.elite_idx()
        elite_chromosomes = deepcopy(chromosomes[elite_idx])
        offspring_size = self.chromosome_size - len(elite_idx)
        select_parents_idx = self.selection.pick_parents(self.num_parents, offspring_size)
        
        select_parents_idx = select_parents_idx.T.flatten()
        parents = chromosomes[select_parents_idx].reshape(offspring_size, 4, -1)

        
        offspring = self.crossover(parents)
        offspring = self.mutation(offspring)

        # print(offspring.shape)

        chromosomes = torch.concat([elite_chromosomes, offspring])
        chromosomes = chromosomes.squeeze(dim=0)
        
        self.update_chromosomes(chromosomes, base_ch_shape, after_ch_shape, device)

