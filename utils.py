import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu due to overhead of moving data from cpu to gpu

# Wrapper class for transitions
class Transition():
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def unpack(self):
        return self.state, self.action, self.reward, self.next_state, self.done