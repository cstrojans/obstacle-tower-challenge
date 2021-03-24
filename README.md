# Obstacle Tower Challenge

## Goal
To create an autonomous AI agent that can play the [Obstacle Tower Challenge](https://unity3d.com/otc) game and climb to the highest level possible.

## Setup Instructions
1. Install python 3.8.0 in your machine using [pyenv](https://github.com/pyenv/pyenv)
2. Fork the repository from [here](https://github.com/cstrojans/obstacle-tower-challenge.git).
3. Clone the repositoy from your Github profile
```bash
git clone https://github.com/<YOUR_USERNAME>/obstacle-tower-challenge.git
```
4. Run the following commands:
```bash
cd obstacle-tower-challenge/

# Set python version for the local folder
pyenv local 3.8.0

# Install pyenv-virtualenv
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
source ~/.bashrc
mkdir venv
cd venv/
pyenv virtualenv 3.8.0 venv
cd ..

# activate virtual environment
pyenv activate venv

# confirm python version
python -V

# Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

```
5. Setup jupyter to work with the [virtual environment](https://albertauyeung.github.io/2020/08/17/pyenv-jupyter.html)
6. By default, the binary will be automatically downloaded when the Obstacle Tower gym is first instantiated. The following line in the Jupyter notebook instantiates the environment:
```
env = ObstacleTowerEnv(retro=False, realtime_mode=False)
```
7. The binaries for each platform can be separately downloaded at the following links. Using these binaries you can play the game.

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_windows.zip |


## Game details

The environment provided has a MultiDiscrete action space (list of valid actions), where the 4 dimensions are: MultiDiscrete([3 3 2 3])
0. Movement (No-Op/Forward/Back)
1. Camera Rotation (No-Op/Counter-Clockwiseorward/Ba/Clockwise)
2. Jump (No-Op/Jump)
3. Movement (No-Op/Right/Left)

The observation space provided includes a 168x168 image (the camera from the simulation) as well as the number of keys held by the agent (0-5) and the amount of time remaining.

## Models and their usage
1. Random Agent
```
usage: train.py random [-h] [--max-eps MAX_EPS] [--save-dir SAVE_DIR]

optional arguments:
  -h, --help           show this help message and exit
  --max-eps MAX_EPS    Maximum number of episodes (games) to run.
  --save-dir SAVE_DIR  Directory in which you desire to save the model.
```

2. A3C Agent
```
usage: train.py a3c [-h] [--lr LR] [--max-eps MAX_EPS] [--update-freq UPDATE_FREQ] [--gamma GAMMA] [--num-workers NUM_WORKERS] [--save-dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate for the shared optimizer.
  --max-eps MAX_EPS     Maximum number of episodes (games) to run.
  --update-freq UPDATE_FREQ
                        How often to update the global model.
  --gamma GAMMA         Discount factor of rewards.
  --num-workers NUM_WORKERS
                        Number of workers for asynchronous learning.
  --save-dir SAVE_DIR   Directory in which you desire to save the model.
```
3. Curiosity Agent
```
usage: train.py curiosity [-h] [--lr LR] [--timesteps TIMESTEPS] [--batch-size BATCH_SIZE] [--gamma GAMMA] [--save-dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate for the shared optimizer.
  --timesteps TIMESTEPS
                        Maximum number of episodes (games) to run.
  --batch-size BATCH_SIZE
                        How often to update the global model.
  --gamma GAMMA         Discount factor of rewards.
  --save-dir SAVE_DIR   Directory in which you desire to save the model.
```
### Commands
To train the agent:
```bash
python src/train.py --env <PATH_TO_OTC_GAME> <AGENT_NAME> [<ARGS>]
```
View training logs on Tensorboard:
```
# to view graphs in tensorboard
tensorboard --logdir logs/
```

To play a game with a trained agent:
```
# play an episode of the game using a given policy (random or a3c)
python src/play.py --env <PATH_TO_OTC_GAME> --algorithm random

# evaluate a given agent
python play.py --env <PATH_TO_OTC_GAME> --algorithm random --evaluate
```
