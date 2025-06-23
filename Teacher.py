import numpy as np
from collections import deque

class Teacher():
    def __init__(self, n_tasks, buffer_length):
        self.buffers = deque(maxlen=n_tasks)
        self.score_buffers = deque(maxlen=n_tasks)
        for i in range(n_tasks):
            self.buffers.append(deque(maxlen=buffer_length))
            self.score_buffers.append(deque(maxlen=2))
    
    def update(self, chosen_task: int, reward: float):
        self.score_buffers[chosen_task].append(reward)
        if len(self.score_buffers[chosen_task]) >= 2:
            self.buffers[chosen_task].append(abs(self.score_buffers[chosen_task][1]-self.score_buffers[chosen_task][0]))

    def sample_task(self):
        sampled_rewards = []
        for i in range(len(self.buffers)):
            if len(self.buffers[i]) == 0:
                sampled_rewards.append(1)
                continue
            sampled_rewards.append(self.buffers[i][np.random.randint(len(self.buffers[i]))])
        return np.argmax(sampled_rewards)