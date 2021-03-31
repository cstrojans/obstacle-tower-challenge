import itertools
import numpy as np
import tensorflow as tf
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
from models.common.constants import train_env_reset_config, eval_env_reset_config
class Memory:
    def __init__(self):
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []

    def store(self, action_prob, value, reward):
        self.action_probs_history.append(action_prob)
        self.critic_value_history.append(value)
        self.rewards_history.append(reward)

    def clear(self):
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        self.rewards_history.clear()


class CuriosityMemory:
    def __init__(self):
        self.frames = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_indices = []
        self.policy = []
        self.state_features = []
        self.new_state_features = []

    def store(self, state, reward, done, value, action_one_hot, policy, state_f, new_state_f):
        self.frames.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_indices.append(action_one_hot)
        self.policy.append(policy)
        self.state_features.append(state_f)
        self.new_state_features.append(new_state_f)
    
    def clear(self):
        self.frames.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_indices.clear()
        self.policy.clear()
        self.state_features.clear()
        self.new_state_features.clear()

class Memory_PPO:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.is_terminals = []

        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.action_probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.is_terminals),\
                batches
    
    def store(self, state, action, action_prob, value, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.is_terminals = []


class ActionSpace:
    def __init__(self):
        """__init__ gets the meaning of action in a multi-discrete space from a scalar space
        The environment provided has a MultiDiscrete action space, where the 4 dimensions are:
        0. Movement (No-Op/Forward/Back)
        1. Camera Rotation (No-Op/Counter-Clockwise/Clockwise)
        2. Jump (No-Op/Jump)
        3. Movement (No-Op/Right/Left)
        """
        self._branched_action_space = [3, 3, 2, 3]
        self._possible_vals = [range(_num)
                               for _num in self._branched_action_space]
        self.all_actions = [
            list(_action) for _action in itertools.product(*self._possible_vals)]
        self.actions = [
            {  # movement
                0: "no-op",
                1: "forward",
                2: "backward"
            },
            {  # camera_rotation
                0: "no-op",
                1: "counter-clockwise",
                2: "clockwise"
            },
            {  # Jump
                0: "no-op",
                1: "jump"
            },
            {  # movement
                0: "no-op",
                1: "right",
                2: "left"
            }
        ]

    """
    def get_action_meaning(self, action):
        # with retro=False
        action_permutation = self.all_actions[action]
        return "[{}, {}, {}, {}]".format(
            self.actions[0][action_permutation[0]],
            self.actions[1][action_permutation[1]],
            self.actions[2][action_permutation[2]],
            self.actions[3][action_permutation[3]]
        )
    """

    def get_action_meaning(self, action):
        # with retro=False
        return "[{}, {}, {}, {}]".format(
            self.actions[0][action[0]],
            self.actions[1][action[1]],
            self.actions[2][action[2]],
            self.actions[3][action[3]]
        )


def record(episode, episode_reward, worker_idx, global_ep_reward, result_queue, total_loss, ep_steps, global_steps):
    """prints statistics
    Args:
        episode: Current episode
        episode_reward: Reward accumulated over the current episode
        worker_idx: Which thread (worker)
        global_ep_reward: The moving average of the global reward
        result_queue: Queue storing the moving average of the scores
        total_loss: The total loss accumualted over the current episode
        num_steps: The number of steps the episode took to complete
    """
    print("Episode: {} | Average Reward: {:.3f} | Episode Reward: {:.3f} | Loss: {:.3f} | Steps: {} | Total Steps: {} | Worker: {}".format(
        episode, global_ep_reward, episode_reward, total_loss, ep_steps, global_steps, worker_idx))
    result_queue.put((global_ep_reward, total_loss))


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def instantiate_environment(path, train, evaluate, eval_seeds=[1001]):
    env = None
    if train:
        env = ObstacleTowerEnv(
            path, worker_id=0, retro=False, realtime_mode=False, greyscale=False, config=train_env_reset_config)
    else:
        if evaluate:
            env = ObstacleTowerEnv(
                path, worker_id=0, retro=False, realtime_mode=False, greyscale=False, config=eval_env_reset_config)
            env = ObstacleTowerEvaluation(env, eval_seeds)
        else:  # play a single game
            env = ObstacleTowerEnv(
                path, worker_id=0, retro=False, realtime_mode=True, greyscale=False, config=eval_env_reset_config)
    
    return env