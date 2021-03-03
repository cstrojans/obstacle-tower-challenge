import itertools
import numpy as np
import tensorflow as tf

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards = []

    def store(self, state, action, action_prob, value, reward):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs_history.append(action_prob)
        self.critic_value_history.append(value)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        self.rewards.clear()


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


def record(episode, episode_reward, worker_idx, global_ep_reward, result_queue, total_loss, num_steps, num_global_steps):
    """Helper function to store score and print statistics.
    Args:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        # f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Loss: {total_loss[0]} | "
        f"Steps: {num_steps} | "
        f"Total Steps: {num_global_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

