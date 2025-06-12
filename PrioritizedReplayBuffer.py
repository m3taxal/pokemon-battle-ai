import random
import numpy as np
from utils import Transition

class PrioritizedReplayBuffer:
    def __init__(self, maxlen, seed=None):
        self.memory = np.full(maxlen, None)
        self.priorities = np.full(maxlen, None)
        self.next_mem_pointer = 0
        self.full_mem_avail = False
        self.maxlen = maxlen

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition: Transition):
        self.memory[self.next_mem_pointer] = transition
        
        # Since this is a new transition, it should receive a high weight be default.
        if not self.full_mem_avail:
            self.priorities[self.next_mem_pointer] = max(self.priorities[:self.next_mem_pointer], default=1)
        else:
            self.priorities[self.next_mem_pointer] = max(self.priorities, default=1)    

        self.next_mem_pointer += 1

        if self.next_mem_pointer >= self.maxlen:
            self.next_mem_pointer = 0
            self.full_mem_avail = True # Now full memory is available and we don't have to slice anymore.

    def get_probabilities(self, priority_scale) -> list[float]:
        if self.full_mem_avail:
            # 1 -> full priority sampling, 0 -> uniform random sampling.
            scaled_priorities = self.priorities**priority_scale
        else:
            scaled_priorities = self.priorities[:self.next_mem_pointer]**priority_scale

        # Normalize priorities.
        sample_probabilities = scaled_priorities / np.sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities, beta) -> list[float]:
        # Apply formula from PER to calculate importance weights, e.g.
        # (1/size_of_buffer * 1/priority)**beta for every priority.
        if self.full_mem_avail:
            importance = (1/self.maxlen * 1/probabilities)**beta
        else:
            importance = (1/self.next_mem_pointer * 1/probabilities)**beta    

        # Normalize importance weights.
        importance_normalized = importance / np.max(importance)
        return importance_normalized

    def sample(self, sample_size, priority_scale, beta) -> tuple[list[Transition.unpack], np.array, list[int]]:
        sample_probs = self.get_probabilities(priority_scale)
        if self.full_mem_avail:
            sample_indices = random.choices(range(self.maxlen), k=sample_size, weights=sample_probs)
        else:
            sample_indices = random.choices(range(self.next_mem_pointer), k=sample_size, weights=sample_probs)    
        samples = [self.memory[i].unpack() for i in sample_indices]
        importance = self.get_importance(sample_probs[sample_indices], beta)
        return samples, np.array(importance, dtype=np.float32), sample_indices

    def update_probabilites(self, indices, errors, offset):
        for i, e in zip(indices, errors):
            # We add an offset to our priorities because the
            # td errors can be 0.
            self.priorities[i] = e.item() + offset

    def __len__(self):
        if self.full_mem_avail:
            return self.maxlen
        else:
            return self.next_mem_pointer