import itertools
import tensorflow as tf

class FrameProcessor(object):
    """Resizes and converts RGB Atari frames to grayscale"""

    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(
            self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed,
                                                [self.frame_height,
                                                    self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def __call__(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame: frame})


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

    def get_action_meaning(self, action):
        action_permutation = self.all_actions[action]
        return "[{}, {}, {}, {}]".format(
            self.actions[0][action_permutation[0]],
            self.actions[1][action_permutation[1]],
            self.actions[2][action_permutation[2]],
            self.actions[3][action_permutation[3]]
        )


def record(episode, episode_reward, worker_idx, global_ep_reward, result_queue, total_loss, num_steps):
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
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward
