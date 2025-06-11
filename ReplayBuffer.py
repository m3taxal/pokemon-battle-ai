from collections import deque
import random

class ReplayBuffer:
    def __init__(self, size: int, batch_size: int, n_step: int, gamma: float):
        self.memory = deque([], maxlen=size)
        self.batch_size = batch_size
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, transition):
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return False
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info()
        obs, act = self.n_step_buffer[0][:2]
        
        self.memory.append((obs, act, rew, next_obs, done))

        return True

    def sample(self, sample_size: int):
        return random.sample(self.memory, sample_size)
    
    def sample_from_indices(self, indices):
        return [self.memory[i] for i in indices]
    
    def _get_n_step_info(self):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = self.n_step_buffer[-1][2:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[2:]

            rew = r + self.gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return len(self.memory)