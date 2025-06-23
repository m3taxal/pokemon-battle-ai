import random
from utils import Transition
import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple

class ReplayBuffer:
    def __init__(self, size: int, batch_size: int):
        self.size = size
        self.memory = np.full(size, None)
        self.next_memory_pointer = 0
        self.batch_size = batch_size
        self.full_mem_avail = False

    def store(self, transition: Transition) -> bool:
        self.memory[self.next_memory_pointer] = transition
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
        return [self.memory[i].unpack() for i in indices]
    
    def __len__(self) -> int:
        if self.full_mem_avail:
            return self.size
        else:
            return self.next_memory_pointer