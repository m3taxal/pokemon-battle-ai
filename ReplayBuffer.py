import random
from utils import Transition
import numpy as np

class ReplayBuffer:
    def __init__(self, size: int, batch_size: int, n_step: int, gamma: float):
        self.size = size
        self.memory = np.full(size, None)
        self.next_memory_pointer = 0
        self.batch_size = batch_size
        self.full_mem_avail = False
        
        # Used for n-step Learning.
        self.n_step_buffer = np.full(n_step, None)
        self.next_n_step_pointer = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_gammas = [gamma**i for i in range(1, n_step)] # So we don't have to calculate powers of gamma every time.
        self.full_n_step_avail = False

    def store(self, transition: Transition) -> bool:
        self.n_step_buffer[self.next_n_step_pointer] = transition
        self.next_n_step_pointer += 1

        if self.next_n_step_pointer >= self.n_step:
            self.next_n_step_pointer = 0
            self.full_n_step_avail = True # Our n_step buffer is filled and we can start
                                          # calculating our n-step transitions.

        # Single step transition is not ready yet.
        if not self.full_n_step_avail:
            return False
        
        # Make a n-step transition.
        rew, next_obs, done = self._get_n_step_info()
        obs, act = self.n_step_buffer[0].unpack()[:2]
        
        self.memory[self.next_memory_pointer] = (Transition(obs, act, rew, next_obs, done))
        self.next_memory_pointer += 1

        if self.next_memory_pointer >= self.size:
            self.next_memory_pointer = 0
            self.full_mem_avail = True  # Now we can use full memory and don't have to slice.
        return True

    def sample(self, sample_size: int) -> list[Transition]:
        if self.full_mem_avail:
            return random.sample(self.memory, sample_size)
        else:
            return random.sample(self.memory[self.next_memory_pointer], sample_size)
    
    def sample_from_indices(self, indices) -> list[Transition.unpack]:
        return [self.memory[i].unpack() for i in indices]
    
    def _get_n_step_info(self) -> tuple[float, np.array, int]:
        # We want to calculate the n-step reward, next_obs and done for
        # the first transition in the buffer.
        first_transition: Transition = self.n_step_buffer[0]
        rew, next_obs, done = first_transition.reward, first_transition.next_state, first_transition.done

        for i in range(1, self.n_step):
            if done:
                # If we are in a terminal state we cannot calculate the n-step anymore,
                # so we stop.
                break
            # We want to go to the next transition.
            next_transition: Transition = self.n_step_buffer[i]

            # Add discounted reward.
            rew += self.n_step_gammas[i-1] * next_transition.reward
            
            next_obs = next_transition.next_state
            done = next_transition.done
        
        return rew, next_obs, done

    def __len__(self) -> int:
        if self.full_mem_avail:
            return self.size
        else:
            return self.next_memory_pointer