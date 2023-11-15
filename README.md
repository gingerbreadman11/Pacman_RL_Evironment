# PacmanDQN
Deep Reinforcement Learning in Pac-man with convolutional DQNs

## Example usage

Run a model on `smallGrid` layout for 6000 episodes, of which 5000 episodes
are used for training.

```
python3 pacman.py -p PacmanDQN -n 6000 -x 5000 -l smallGrid

or loading a from a pre trained NN (a pre trained DQN only works on maps that are the same size as the original map that was used for training)

python3 pacman.py -p PacmanDQN -a load_file=saves/model-5layer_grid_2397419_151613 -n 102 -x 10 -l mediumGrid4

python3 plotmaker.py can be used for making plots based on the logs 
```

### Layouts
Different layouts can be found and created in the `layouts` directory

### Parameters

Parameters can be found in the `params` dictionary in `pacmanDQN_Agents.py`. <br />
 <br />
Models are saved as "checkpoint" files in the `/saves` directory. <br />
Load and save filenames can be set using the `load_file` and `save_file` parameters. <br />
 <br />
Episodes before training starts: `train_start` <br />
Size of replay memory batch size: `batch_size` <br />
Amount of experience tuples in replay memory: `mem_size` <br />
Discount rate (gamma value): `discount` <br />
Learning rate: `lr` <br />
 <br />
Exploration/Exploitation (ε-greedy): <br />
Epsilon start value: `eps` <br />
Epsilon final value: `eps_final` <br />
Number of steps between start and final epsilon value (linear): `eps_step` <br />

to change the DQN change it in pacmanDQN_Agents.py
by changing the following on line 69:
        
        #here you can choose the DQN you want to run by changing the number to 2,3,4,5
        #self.qnet = DQN4(self.params)

        # Start Tensorflow session
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
        self.qnet = DQN4(self.params)

## Acknowledgements

thease repository were used for implementing the code:

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))
van der Ouderaa, Tycho (2016). Deep Reinforcement Learning in Pac-man.
