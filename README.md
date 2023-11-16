# PacmanDQN
Deep Reinforcement Learning in Pac-man using convolutional Deep Q-Networks (DQNs).

## Running the Game

### Basic Usage
Run a model on a specific layout for a set number of episodes:
```bash
python3 pacman.py -p PacmanDQN -n [num_episodes] -x [num_training] -l [layout]
```

### Using Pre-trained Models
Load a pre-trained model (note: only compatible with layouts identical in size to the one used during training):
```bash
python3 pacman.py -p PacmanDQN -a load_file=[model_path] -n [num_episodes] -x [num_training] -l [layout]
```

#### Compatibility of Pre-trained Models

It's crucial to note that pre-trained models are only compatible with game layouts that are identical in size to the ones used during their training. Additionally, the number of layers in the DQN must match between the training and deployment phases. This means that a model trained with a certain number of layers will only function correctly on a network with the same layer configuration.


```bash
# Using a 2-layer model on mediumGrid layout for 5 episodes
python3 pacman.py -p PacmanDQN -n 5 -x 1 -l mediumGrid -a load_file=saves/model-2layer_967413_31617

# Switching to a 4-layer model
python3 pacman.py -p PacmanDQN -n 5 -x 1 -l mediumGrid -a load_file=saves/model-4layer_grid_920590_28440

# For the ultimate model on mediumClassic layout
python3 pacman.py -p PacmanDQN -n 5 -x 1 -l mediumClassic -a load_file=saves/model-ultimate_1084668_5469
```
### pre trained models / DQNs / maps that are compatible:
model-2layer_967413_31617               ->      DQN2    ->      mediumGrid, mediumGrid2, mediumGrid3, mediumGrid4, mediumGrid5 
model-3layer_711729_37302               ->      DQN3    ->      mediumGrid, mediumGrid2, mediumGrid3, mediumGrid4, mediumGrid5 
model-4layer_grid_920590_28440          ->      DQN4    ->      mediumGrid, mediumGrid2, mediumGrid3, mediumGrid4, mediumGrid5 
model-5layer_grid_2443870_155162        ->      DQN5    ->      mediumGrid, mediumGrid2, mediumGrid3, mediumGrid4, mediumGrid5 
model-ultimate_1084668_5469             ->      DQN4    ->      mediumClassic, nowalls, noghosts, easy

### Commands Breakdown
- `-p PacmanDQN`: Specifies the use of the PacmanDQN agent.
- `-n`: Number of total episodes to run.
- `-x`: Number of episodes for training before evaluation.
- `-l`: Specifies the layout of the game (e.g., `mediumGrid`).
- `-a load_file`: Path to load the pre-trained model.

## Visualizing Results
Use `python3 plotmaker.py` to create plots from log data.
to visualize a specific plot you have to change the path in the file plotmaker.py

## Layouts
Different layouts are available in the `layouts` directory. New layouts can be created and added to this directory.

you can easily creat eyour own maps just make sure its a lay file
that is the same size as the map that the model was trained on

## Customizing Parameters
Parameters are found in the `params` dictionary in `pacmanDQN_Agents.py`. This includes:
- `train_start`: Episodes before training starts.
- `batch_size`: Size of replay memory batch.
- `mem_size`: Amount of experience tuples in replay memory.
- `discount`: Discount rate (gamma value).
- `lr`: Learning rate.
- Exploration/Exploitation settings (ε-greedy):
    - `eps`: Epsilon start value.
    - `eps_final`: Epsilon final value.
    - `eps_step`: Steps between start and final epsilon value.

### Important Configuration Information

#### Changing the Number of Layers in the DQN

To adjust the layers of the Deep Q-Network (DQN), you need to modify the specific DQN function that is invoked in the code. This change determines the number of layers in the neural network, thus affecting its learning and performance capabilities.

**Location and Steps for Modification:**

1. **File**: `pacmanDQN_Agents.py`
2. **Line**: 81
3. **Modification**: Change the `DQN` function call to specify the desired number of layers.
    - **Example**: To change from a 4-layer network to a network with a different number of layers, modify the line as follows:

      Before:
      ```python
      self.qnet = DQN4(self.params)
      ```
      After (for a 5-layer network, for example):
      ```python
      self.qnet = DQN5(self.params)
      ```

## Acknowledgements
This implementation utilizes resources from the following:

- DQN Framework adapted for Pac-man (originally developed for ATARI games):
  * [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow)

- Pac-man implementation by UC Berkeley:
  * [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html)

Referenced in Tycho van der Ouderaa's 2016 work, "Deep Reinforcement Learning in Pac-man."