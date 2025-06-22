import random
from utils import Transition
import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple

class ReplayBuffer:
    def __init__(self, size: int, batch_size: int, n_step: int, gamma: float):
        self.size = size
        self.memory = np.full(size, None)
        self.next_memory_pointer = 0
        self.batch_size = batch_size
        self.full_mem_avail = False

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, transition: Transition) -> bool:
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return False
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0].unpack()[:2]

        self.memory[self.next_memory_pointer] = Transition(obs, act, rew, next_obs, done)
        self.next_memory_pointer += 1

        if self.next_memory_pointer >= self.size:
            self.next_memory_pointer = 0
            self.full_mem_avail = True  # Now we can use full memory and don't have to slice.
        return True

    def sample(self) -> list[Transition.unpack]:
        if self.full_mem_avail:
            indices = np.random.randint(low=self.size, size=self.batch_size)
        else:
            indices = np.random.randint(low=self.next_memory_pointer, size=self.batch_size)
        return [self.memory[i].unpack() for i in indices], indices
        
    def sample_from_idxs(self, indices):
        return [self.memory[i].unpack() for i in indices]

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1].unpack()[-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition.unpack()[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        if self.full_mem_avail:
            return self.size
        else:
            return self.next_memory_pointer