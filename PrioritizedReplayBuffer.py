import random
from collections import deque
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, maxlen, seed=None):
        self.memory = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities)**priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities, beta):
        importance = (1/len(self.memory) * 1/probabilities)**beta
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, sample_size, priority_scale, beta):
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.memory)), k=sample_size, weights=sample_probs)
        samples = [self.memory[i] for i in sample_indices]
        importance = self.get_importance(sample_probs[sample_indices], beta)
        return samples, importance, sample_indices

    def update_probabilites(self, indices, errors, offset):
        for i, e in zip(indices, errors):
            self.priorities[i] = e.item() + offset

    def __len__(self):
        return len(self.memory)