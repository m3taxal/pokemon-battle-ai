import random
import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from PokemonBattleEnv import PokemonBattleEnv
from agent_dqn import DQN
from ReplayBuffer import ReplayBuffer
from utils import device, Transition
import torch.nn.functional as F
from Teacher import Teacher
from collections import deque

# Directory for saving run info.
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMETERS = os.path.join(MAIN_DIR, "hyperparameters.yml")
RUNS_DIR = os.path.join(MAIN_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# For printing date and time.
DATE_FORMAT = "%m-%d %H:%M:%S"

# Deep Q-Learning Agent.
class Agent():

    def __init__(self, hyperparameter_set='pokemon_battle'):
        with open(PARAMETERS, 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            self.hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        # Hyperparameters (adjustable).
        self.task_buffer            = self.hyperparameters['task_buffer']            # how many last rewards should be remembered for each opponent for sampling
        self.task_length            = self.hyperparameters['task_length']            # how long a single task should be, i.e. how many battles should be fought
        self.replay_memory_size     = self.hyperparameters['replay_memory_size']     # size of replay memory
        self.num_quant              = self.hyperparameters['num_quant']              # how many quantiles should be estimated
        self.mini_batch_size        = self.hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.network_sync_rate      = self.hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.learning_rate_a        = self.hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g      = self.hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.train_freq             = self.hyperparameters['train_freq']             # update model every train_freq steps
        self.steps                  = self.hyperparameters['steps']                  # how many steps model should train for
        self.epsilon_init           = self.hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_fraction       = self.hyperparameters['epsilon_fraction']       # over what fraction of steps epsilon should be decayed until epsilon_min
        self.epsilon_min            = self.hyperparameters['epsilon_min']            # minimum epsilon value
        self.randomize_enemy        = self.hyperparameters['randomize_enemy']        # if opponent should be randomized each battle
        self.log_freq               = self.hyperparameters['log_freq']               # how often we should log our results (in steps)
        self.continue_training      = self.hyperparameters['continue_training']      # whether or not to continue training previous model (False if no previous model)

        # Quantile tau for loss calculation.
        self.quantile_tau = (torch.arange(self.num_quant, dtype=torch.float32) + 0.5) / self.num_quant
        self.quantile_tau = self.quantile_tau.to(device).view(1, self.num_quant)
        
        self.optimizer = None # NN Optimizer. Initialize later.

        # Path to Run info.
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        # Create instance of the environment.
        self.env = PokemonBattleEnv(self.randomize_enemy)

        # Init teacher.
        self.teacher = Teacher(len(self.env.opponents), self.task_buffer)

    def normalize_rew(self, rew):
        return rew / self.env.max_reward

    def train(self):
        # Log initial hyperparameters.
        with open(self.LOG_FILE, 'w') as file:
            file.write("Logging hyperparameters..." + "\n")
            for key, value in self.hyperparameters.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")

        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Beginning new training session... " + "\n"
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')

        num_actions = self.env.action_space.n

        # Create policy network.
        policy_dqn = DQN(action_dim=num_actions, num_quantiles=self.num_quant).to(device)
        
        # We want to keep training the same network and not start over.
        if self.continue_training:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

        # Initialize epsilon.
        epsilon = self.epsilon_init
        
        # Initialize normal memory with n_step = 1.
        memory = ReplayBuffer(self.replay_memory_size, self.mini_batch_size)

        # Create the target network and make it identical to the policy network.
        target_dqn = DQN(action_dim=num_actions, num_quantiles=self.num_quant).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        state, _ = self.env.reset()  # Initialize environment. Reset returns (state,info).    
        state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device.

        # Teacher attributes.
        episode_reward = 0
        task_reward = np.array([])
        chosen_task = self.teacher.sample_task()
        self.env.set_opponent(chosen_task)

        for step in range(self.steps):
            # Select action based on epsilon-greedy.
            if random.random() < epsilon:
                # select random action
                action = torch.tensor(self.env.action_space.sample(), dtype=torch.int64, device=device)
            else:
                # select best action
                with torch.no_grad():
                    q_dist = policy_dqn(state.unsqueeze(0))  # shape: (1, action_dim, num_quant)
                    q_values = q_dist.mean(dim=2)  # expected values
                    action = q_values.squeeze().argmax()
                    
            # Execute action. Truncated and info is not used.
            new_state, reward, terminated, truncated, info = self.env.step(action.item())

            episode_reward += reward

            # Convert new state and reward to tensors on device.
            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)

            # Store transition into memory.
            memory.store(Transition(state, action, reward, new_state, int(terminated)))

            # Move to the next state.
            state = new_state

            # If enough experience has been collected.
            if len(memory)>=self.mini_batch_size:
                # Update every train_freq.
                if step % self.train_freq == 0:
                    self.optimize(memory.sample(), policy_dqn, target_dqn)

                if self.steps % self.network_sync_rate == 0:
                    # Copy policy network to target network.
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            # Save model and log results if another log_freq steps have been processed.
            if step % self.log_freq == 0 and step > 0: # No use in logging the 0th step.
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                # Logs metrics
                log_message = self.log(datetime.now().strftime(DATE_FORMAT), step, self.teacher.score_buffers, epsilon)
                print(log_message + "\n")
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n'*2)

            if terminated:
                # Linear epsilon decay over fraction of training episodes.
                if epsilon > self.epsilon_min:
                    epsilon -= (self.epsilon_init-self.epsilon_min) / (self.epsilon_fraction*self.task_length)

                    # Possible floating point inaccuracies could occur, so set epsilon to epsilon_min if necessary.
                    if epsilon < self.epsilon_min:
                        epsilon = self.epsilon_min

                task_reward = np.append(task_reward, self.normalize_rew(episode_reward))

                # We have fought self.task_length battles.
                # Update teacher, choose new task and reset teacher attributes.
                if len(task_reward)==self.task_length:
                    print("Began a new task.\n")
                    self.teacher.update(chosen_task, np.mean(task_reward[int(self.epsilon_fraction*len(task_reward)):]))
                    chosen_task = self.teacher.sample_task()
                    self.env.set_opponent(chosen_task)
                    task_reward = np.array([])
                    epsilon = self.epsilon_init
                    
                # Reset environment for next episode. Reset returns (state,info).    
                state = torch.tensor(self.env.reset()[0], dtype=torch.float, device=device)

                # Reset episode attributes for next episode.
                episode_reward = 0
        
        print("Training finished!")

    def log(self, time: str, step: int, task_rew, epsilon) -> str:
            lines = [
                f"Logged at {time}",
                f"Step: {step}"
            ]

            for i in range(len(task_rew)):
                new_line = f"Task {i} last 2 average rewards: "
                for rew in task_rew[i]:
                    new_line += str(rew) + ", "
                lines.append(new_line)

            lines.append(f"Current epsilon: {epsilon:.4f}")

            return "\n".join(lines)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Calculate loss of n-step transition batch.
        loss = self.compute_loss(mini_batch, policy_dqn, target_dqn, self.discount_factor_g)

        # Optimize the model (backpropagation).
        self.optimizer.zero_grad()  # Clear gradients.
        loss.backward()             # Compute gradients.
        self.optimizer.step()       # Update network parameters i.e. weights and biases.

    def compute_loss(self, mini_batch, policy_dqn, target_dqn, gamma):
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.stack(states)                     # (batch, state_dim)
        actions = torch.stack(actions).unsqueeze(1)      # (batch, 1)
        rewards = torch.stack(rewards).unsqueeze(1)      # (batch, 1)
        next_states = torch.stack(next_states)           # (batch, state_dim)
        dones = torch.tensor(dones).float().unsqueeze(1).to(device)  # (batch, 1)

        batch_size = states.size(0)
        N = self.num_quant

        # --- Compute target quantile distribution ---
        with torch.no_grad():
            next_q_values = policy_dqn(next_states).mean(dim=2)  # (batch, action_dim)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)  # (batch, 1)

            next_quantiles = target_dqn(next_states)  # (batch, action_dim, num_quant)
            next_quantiles = next_quantiles.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, N))  # (batch, 1, N)
            next_quantiles = next_quantiles.squeeze(1)  # (batch, N)

            target_quantiles = rewards + (1 - dones) * gamma * next_quantiles  # (batch, N)
            target_quantiles = target_quantiles.unsqueeze(1)  # (batch, 1, N)

        # --- Predicted quantiles ---
        all_quantiles = policy_dqn(states)  # (batch, action_dim, N)
        chosen_quantiles = all_quantiles.gather(1, actions.unsqueeze(-1).expand(-1, -1, N))  # (batch, 1, N)

        # --- TD error (used for weighting)
        td_error = target_quantiles.unsqueeze(1) - chosen_quantiles.unsqueeze(2)  # (batch, N, N)
        
        # --- Huber loss with correct shape
        huber_loss = F.smooth_l1_loss(
            chosen_quantiles.unsqueeze(2),     # (batch, N, 1)
            target_quantiles.unsqueeze(1),     # (batch, 1, N)
            reduction='none'
        )  # (batch, N, N)

        # --- Quantile weights
        tau = self.quantile_tau.to(device).view(1, N).expand(batch_size, N).unsqueeze(2)  # (batch, N, 1)
        weight = torch.abs((td_error.detach() < 0).float() - tau)  # (batch, N, N)

        loss = (weight * huber_loss).sum(dim=2).mean(dim=1).mean()
        return loss

    def test(self, randomized_enemy: bool = False, battles_to_play: int = 1000):
        print("Starting evaluation...")

        # Create instance of the environment.
        env = PokemonBattleEnv(randomized_enemy)

        # Create DQN agent.
        dqn = DQN(action_dim=env.action_space.n, num_quantiles=self.num_quant).to(device)

        # Load trained model.
        dqn.load_state_dict(torch.load(self.MODEL_FILE))
        dqn.eval()

        # Initialize evaluation.
        state, _ = env.reset()   
        state = torch.tensor(state, dtype=torch.float, device=device)

        for battle in range(battles_to_play):
            terminated = False
            
            while not terminated:
                # We act 100% greedy in real battles.
                q_dist = dqn(state.unsqueeze(0))
                q_values = q_dist.mean(dim=2)
                action = q_values.squeeze().argmax()

                new_state,reward,terminated,truncated,info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                state = new_state # Proceed to next state

            state, _ = env.reset() # Reset if battle finished
            state = torch.tensor(state, dtype=torch.float, device=device)

        print(f"Evaluation finished after {battles_to_play} battles. Achieved winrate of {env.winrate:.3f}.")

    def save_graph(self, episode_reward_history, winrate_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(episode_reward_history))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(episode_reward_history[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.ylabel('Winrate')
        plt.plot(winrate_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--test', help='Testing mode', action='store_true')
    args = parser.parse_args()

    dql = Agent()

    if args.train and not args.test:
        dql.train()
    
    if args.test and not args.train:
        dql.test()