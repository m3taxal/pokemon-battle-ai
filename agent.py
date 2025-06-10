import random
import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from PokemonBattleEnv import PokemonBattleEnv
from experience_replay import ReplayMemory
from dqn import DQN
from LogWindow import LogWindow
from prio_replay import PrioritizedReplayBuffer
from custom_encodings import ENCODING_CONSTANTS
from utils import device

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
        self.env_id             = self.hyperparameters['env_id']                 # used for gym.make()
        self.replay_memory_size = self.hyperparameters['replay_memory_size']     # size of replay memory
        self.replay_offset      = self.hyperparameters['replay_offset']          # add offset so errors are never 0
        self.replay_alpha       = self.hyperparameters['replay_alpha']           # 1 = full priority sampling, 0 = uniform random sampling
        self.replay_beta        = self.hyperparameters['replay_beta']            # fix bias towards higher priorities in priority sampling
        self.mini_batch_size    = self.hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.network_sync_rate  = self.hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.learning_rate_a    = self.hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = self.hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.train_freq         = self.hyperparameters['train_freq']             # update model every train_freq steps
        self.steps              = self.hyperparameters['steps']                  # how many steps model should train for
        self.epsilon_init       = self.hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_fraction   = self.hyperparameters['epsilon_fraction']       # over what fraction of steps epsilon should be decayed until epsilon_min
        self.epsilon_min        = self.hyperparameters['epsilon_min']            # minimum epsilon value
        self.randomize_enemy    = self.hyperparameters['randomize_enemy']        # if opponent should be randomized each battle
        self.log_freq           = self.hyperparameters['log_freq']               # how often we should log our results
        self.should_log         = self.hyperparameters['should_log']             # to log, or not to log?
        self.plot_freq          = self.hyperparameters['plot_freq']              # how often we should plot our results
        self.should_plot        = self.hyperparameters['should_plot']            # to plot, or not to plot?
        self.continue_training  = self.hyperparameters['continue_training']      # whether or not to continue training previous model (False if no previous model)

        # For logging results.
        self.log_window = LogWindow()

        # Neural Network.
        self.loss_fn = torch.nn.MSELoss() # NN Loss function.
        self.optimizer = None # NN Optimizer. Initialize later.

        # Path to Run info.
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

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

        # Create instance of the environment.
        env = PokemonBattleEnv(self.randomize_enemy)

        num_actions = env.action_space.n

        # Create policy network.
        policy_dqn = DQN(action_dim=num_actions).to(device)

        # We want to keep training the same network and not start over.
        if self.continue_training:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

        # Initialize epsilon.
        epsilon = self.epsilon_init
        
        # Initialize replay memory.
        # TODO: Add tunable hyperparameters
        memory = PrioritizedReplayBuffer(self.replay_memory_size)

        # Create the target network and make it identical to the policy network.
        target_dqn = DQN().to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        state, _ = env.reset()  # Initialize environment. Reset returns (state,info).    
        state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device.

        # For plotting graphs
        episode_reward = 0
        episode_count  = 0
        reward_history = np.array([])

        for step in range(self.steps):
            # Select action based on epsilon-greedy.
            if random.random() < epsilon:
                # select random action
                action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
            else:
                # select best action
                with torch.no_grad():
                    # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                    # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                    # argmax finds the index of the largest element.
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

            # Execute action. Truncated and info is not used.
            new_state, reward, terminated, truncated, info = env.step(action.item())

            episode_reward += reward

            # Convert new state and reward to tensors on device.
            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)

            # Save experience into memory.
            memory.add((state, action, reward, new_state, terminated))

            # Move to the next state.
            state = new_state

            # If enough experience has been collected.
            if len(memory)>self.mini_batch_size:
                # Update every train_freq.
                if step % self.train_freq == 0:
                    mini_batch, weights, tree_idx = memory.sample(self.mini_batch_size)
                    td_error = self.optimize(mini_batch, policy_dqn, target_dqn, weights)

                    memory.update_priorities(tree_idx, td_error.numpy())

                if self.steps % self.network_sync_rate == 0:
                    # Copy policy network to target network.
                    target_dqn.load_state_dict(policy_dqn.state_dict())

                # Linear epsilon decay over fraction of training steps.
                if epsilon > self.epsilon_min:
                    decay_factor = ((self.epsilon_init-self.epsilon_min) / (self.epsilon_fraction*self.steps))
                    epsilon -= decay_factor

                    # Possible floating point inaccuracies could occur, so set epsilon to epsilon_min if necessary.
                    if epsilon < self.epsilon_min:
                        epsilon = self.epsilon_min

            # Save model and log results if another log_freq steps have been processed.
            if step % self.log_freq == 0 and step > 0: # No use in logging the 0th step
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                if self.should_log:
                    # Not exactly sure which metrics to log... winrate is already being plotted,
                    # and epsilon isn't very interesting to log.
                    log_message = self.log_window.log(datetime.now().strftime(DATE_FORMAT), step, env.winrate, epsilon)
                    print(log_message + "\n")
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n'*2)

            if terminated:
                episode_count += 1
                reward_history = np.append(reward_history, episode_reward)
                
                # Reset environment for next episode. Reset returns (state,info).    
                state = torch.tensor(env.reset()[0], dtype=torch.float, device=device)

                # Plot every plot_freq episodes.
                if self.should_plot and episode_count % self.plot_freq == 0:
                    print(f"Plotting results at episode {episode_count}..." + "\n")
                    self.save_graph(reward_history, env.winrate_history)
                
                # Reset episode attributes for next episode.
                episode_reward = 0
        
        # Plot last state of model.
        if self.should_plot:
            print(f"Plotting results at last step..." + "\n")
            self.save_graph(reward_history, env.winrate_history)    
        
        print("Training finished!")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn, weights=None):
        # Transpose the list of experiences and separate each element.
        states, actions, rewards, new_states, terminations = zip(*mini_batch)
 
        # Stack tensors to create batch tensors.
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns).
            best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

            target_q = rewards + (1-terminations) * self.discount_factor_g * \
                            target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
        
        # Calcuate Q values from current policy.
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        if weights is None:
            weights = torch.ones_like(current_q)

        diff = current_q - target_q

        # To update priorities in replay memory
        td_error = torch.abs(diff).detach()

        # Compute loss.
        loss = torch.mean(diff**2 * weights)

        # Optimize the model (backpropagation).
        self.optimizer.zero_grad()  # Clear gradients.
        loss.backward()             # Compute gradients.
        self.optimizer.step()       # Update network parameters i.e. weights and biases.

        return td_error

    def test(self, randomized_enemy: bool = False, battles_to_play: int = 1000):
        print("Starting evaluation...")

        # Create instance of the environment.
        env = PokemonBattleEnv(randomized_enemy)

        # Create DQN agent.
        dqn = DQN(action_dim=env.action_space.n).to(device)

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
                action = dqn(state.unsqueeze(dim=0)).squeeze().argmax()

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