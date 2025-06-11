# pokemon-battle-ai
DQN agent (with various improvements) for the VGC AI Framework pokemon battle simulator:

https://gitlab.com/DracoStriker/pokemon-vgc-engine

## checkpoints
Contains folders of the model at various stages of training.

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