from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

tf.config.run_functions_eagerly(True)


clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        print(rewards[i], values[i].shape, delta.shape)
        gae = delta + gamma * lmbda * masks[i] * gae
        # print('GAE')
        # print(gae)
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward


def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot

def get_model_actor(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    conv1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(
            4, 4), padding='same', activation='relu', data_format='channels_last', input_shape=input_dims)
        # (9, 9, 32)
    conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(
        2, 2), padding='same', activation='relu', data_format='channels_last')
    flatten = Flatten()  # (9 * 9 * 32)
    dense1 = Dense(units=256, activation='relu')

    # policy output layer (Actor)
    policy_logits = Dense(
        output_dims, activation='softmax', name='policy_logits')
    # values_op = Dense(units=1, name='value')


    x = conv1(state_input)
    x = conv2(x)
    x = flatten(x)
    x = dense1(x)

    logits = policy_logits(x)
    # val = values_op(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[logits])
    
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    model.summary()
    return model
 
def get_model_critic(input_dims):
    state_input = Input(shape=input_dims)
    

    conv1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(
            4, 4), padding='same', activation='relu', data_format='channels_last', input_shape=input_dims)
        # (9, 9, 32)
    conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(
        2, 2), padding='same', activation='relu', data_format='channels_last')
    flatten = Flatten()  # (9 * 9 * 32)
    dense1 = Dense(units=256, activation='relu')

    # policy output layer (Actor)
    
    values_op = Dense(units=1, name='value')


    x = conv1(state_input)
    x = conv2(x)
    x = flatten(x)
    x = dense1(x)

    val = values_op(x)

    model = Model(inputs=[state_input],
                  outputs=[val])
    
    model.compile(optimizer=Adam(lr=1e-4), loss=['mse'])
    model.summary()
    return model

if __name__ == '__main__':

    env = ObstacleTowerEnv(retro=True, realtime_mode=True)

    state = env.reset()
    state_dims = env.observation_space.shape
    n_actions = env.action_space.n
    print(state_dims, n_actions)

    dummy_n = np.zeros((1, 1, n_actions))
    dummy_1 = np.zeros((1, 1, 1))

    tensor_board = TensorBoard(log_dir='./logs')

    actor_model = get_model_actor(input_dims=state_dims, output_dims=n_actions)
    critic_model = get_model_critic(input_dims=state_dims)
    ppo_steps = 10
    target_reached = False
    best_reward = 0
    iters = 0
    max_iters = 50

    while not target_reached and iters < max_iters:

        states = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []
        actions_onehot = []
        state_input = None
        itr = 0
        while True:
            itr += 1 
            state_input = K.expand_dims(state, 0)

            action_dist = actor_model.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
            q_value = critic_model.predict([state_input], steps=1)
            print(action_dist.shape, q_value)
            action = np.random.choice(n_actions, p=action_dist[0, :])
            action_onehot = np.zeros(n_actions)
            action_onehot[action] = 1

            observation, reward, done, info = env.step(action)
            print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
            mask = not done

            states.append(state)
            actions.append(action)
            actions_onehot.append(action_onehot)
            values.append(q_value)
            masks.append(mask)
            rewards.append(reward)
            actions_probs.append(action_dist)

            state = observation
            if done:
                break

        q_value = critic_model.predict([state_input], steps=1)
        values.append(q_value)
        
        returns, advantages = get_advantages(values, masks, rewards)
        
        
        states = np.array(states)
        actions_probs = np.array(actions_probs)
        rewards = np.reshape(rewards, newshape=(-1, 1, 1))
        values = np.array(values[:-1])
        

        actions_onehot = np.reshape(actions_onehot, newshape=(-1, n_actions))
        returns = np.array(returns)
        
        actor_loss = actor_model.fit(
            [states, actions_probs, advantages, rewards, values],
            [actions_onehot], verbose=True, shuffle=True, epochs=20,
            callbacks=[tensor_board])
        critic_loss = critic_model.fit([states], [returns], shuffle=True, epochs=20,
                                    verbose=True, callbacks=[tensor_board])

        # avg_reward = np.mean([test_reward() for _ in range(5)])
        # print('total test reward=' + str(avg_reward))
        # if avg_reward > best_reward:
        #     print('best reward=' + str(avg_reward))
        #     actor_model.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
        #     critic_model.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
        #     best_reward = avg_reward
        # if best_reward > 0.9 or iters > max_iters:
        #     target_reached = True
        
        # actor_model.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
        # critic_model.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
        iters += 1
        
        env.reset()

    env.close()

