# pokemon-battle-ai
DQN agent (with various improvements) for the VGC AI Framework pokemon battle simulator:

https://gitlab.com/DracoStriker/pokemon-vgc-engine

## checkpoints
Contains folders of the model, log files and visualisations at various stages of training.

## agent.py
Contains agent logic such as the main training loop and optimizing step. 

## hyperparameters.yml
Defines hyperparameters (such as learning rate, discount, ...) for agent.

## dqn.py
Contains DQN network architecture. We first embed moves, and then feed the concatenated field features and move features into the network.

## PrioritizedReplayBuffer.py
Used for PER.

## ReplayBuffer.py
Used for n-step learning.

## move_encoder.py
Simple neural network for embedding moves.

## custom_encodings.py
Contains encodings for field effects, pokemon and moves.

## PokemonBattleEnv.py
Defines an OpenAI gym enviroment to be used in conjunction with the pokemon battle simulator. Most interesting part is the `step(...)` method, which defines the enviroment's reward function. 

## utils.py
Defines a wrapper class for transitions aswell as the device on which the agent will be trained on.

# Current problems
The actual processing of training steps takes longer the more training steps have been processed. For example, processing 20k steps at the beginnng (step 0) takes maybe 1 minute, while at step 100k it could take up to 3 minutes.

The increase in processing time isn't unusual since the replay memory keeps accumulating transitions, meaning that sampling will take longer and longer. However, I feel like the current increase in processing time is too big.

The checkpoints in `checkpoints/test_series` were done with PER and n-step, every other checkpoint was done without PER and n-step. `checkpoints/old` can be ignored since a slightly modified DQN architecture was used (and they contained implementation errors, which renders them invalid for comparisons).