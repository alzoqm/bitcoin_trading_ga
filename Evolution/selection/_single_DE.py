import numpy as np
import torch

class SingleObjectiveSelection:
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
        """Check if fitness is single-objective (1-D array)."""
        if self.__single_objective_checked:
            pass
        else:
            msg = 'Need 1-D fitness for Single-Objective Selection Algorithm'
            assert fitness.ndim == 1, msg
            self.__single_objective_checked = True

class DESelection(SingleObjectiveSelection):
    def __init__(
        self, 
        F: float = 0.5,       # Mutation factor
        CR: float = 0.9,      # Crossover probability
        minimize: bool = True
    ):
        self.F = F
        self.CR = CR
        self.minimize = minimize

    def select(self, population: torch.Tensor, fitness: torch.Tensor):
        """Initialize population and fitness."""
        self._check_single_objective(fitness)
        self.population = population  # Shape: (pop_size, num_params)
        self.fitness = fitness        # Shape: (pop_size,)
        self.pop_size = population.shape[0]
        self.num_params = population.shape[1]

        # Determine best individual
        if self.minimize:
            self.__best_idx = torch.argmin(self.fitness)
        else:
            self.__best_idx = torch.argmax(self.fitness)

    def elite_idx(self):
        """Return the index of the best individual."""
        return self.__best_idx.item()

    def sort_idx(self):
        """Return indices sorted by fitness."""
        if self.minimize:
            return torch.argsort(self.fitness)
        else:
            return torch.argsort(self.fitness, descending=True)

    def pick_parents(self, n: int, len_offspring: int):
        """
        For DE, generate mutant vectors for the entire population.
        Since DE does not use parent selection in the traditional sense,
        this function generates trial vectors for all individuals.
        """
        assert len_offspring == self.pop_size, \
            "len_offspring must be equal to the population size in DE."

        trial_population = torch.zeros_like(self.population)
        indices = np.arange(self.pop_size)

        for i in range(self.pop_size):
            idxs = indices[indices != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, size=3, replace=False)
            a = self.population[a_idx]
            b = self.population[b_idx]
            c = self.population[c_idx]

            # Mutation
            v = a + self.F * (b - c)

            # Crossover
            u = torch.clone(self.population[i])
            jrand = np.random.randint(self.num_params)
            for j in range(self.num_params):
                if np.random.rand() < self.CR or j == jrand:
                    u[j] = v[j]
                # else u[j] remains the same (from target vector)

            trial_population[i] = u

        return trial_population  # Return the trial population

    def best_one(self) -> int:
        """Return the index of the best individual."""
        return self.elite_idx()