# pokemon-battle-ai
Double DQN agent for pokemon battles

## checkpoints
Contains folders of the model at various stages of training. The .pt files
are the saved PyTorch models. The graphs show the mean reward per episode (one episode = one whole pokemon battle, from start to finish) and the epsilon at each time step.

## agent.py
Contains agent logic such as the main training loop and optimizing step. 

## hyperparameters.yml
Defined hyperparamters (such as learning rate, discount, ...) for agent.

## dqn.py
Contains DQN network architecture. We first embed moves, and then feed the concatenated field features and move features into the network.

## experience_replay.py
Used in `dqn.py`.

## move_encoder.py
Simple neural network for encoding moves.

## custom_encodings.py
Contains encodings for field effects, pokemon and moves.

## PokemonBattleEnv.py
Defined an OpenAI gym enviroment to be used in conjunction with the pokemon battle simulator. Most interesting part is the `step(...)` method, which defined the enviroment's reward function. 