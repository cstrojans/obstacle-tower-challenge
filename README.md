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

## Usage
```bash
cd src/

# play an episode of the game using a given policy
python play.py -env <PATH_TO_OTC_GAME> -policy random

# evaluate a given policy
python play.py -env <PATH_TO_OTC_GAME> -policy random -eval

# train an agent

```