import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
from qlearner import QLearner

ACTION_MAP = ['<', 'V', '>', '^']

def get_mdptbx_env_from_gym_env(env):
    T = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n) )
    R = np.zeros((env.observation_space.n))
    for state in env.env.P.keys():
        choices = env.env.P[state]
        for action in choices.keys():
            outcomes = choices[action]
            for outcome in outcomes:
                prob, next_state, reward, terminal = outcome
                T[action][state][next_state] += prob
                if not terminal or state != next_state:
                    R[next_state] = reward
    return T, R

def map_policy_to_action_chars(P):
    return np.array(list(map(lambda a: ACTION_MAP[a], P)))

def plot_policy(P, V, shape=None):
    """
    Params
    ------
    - P (tuple): Policy
    - V (array): Value function
    """
    policy = np.array(P)
    values = np.array(V)
    if shape is None:
        side_size = int(np.sqrt(policy.shape[0]))
        shape = (side_size, side_size)
    policy_chars = map_policy_to_action_chars(policy)
    policy_for_plot = policy_chars.reshape(shape)
    value_for_plot = values.reshape(shape)
    sns.heatmap(value_for_plot, annot=policy_for_plot, fmt='s')
    plt.show()


def train_q_learner(env, epochs=100, max_iters=100, **kws):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    ql = QLearner(num_states, num_actions, )
    start = time.time()
    epoch_steps = np.zeros(epochs)
    epoch_rewards = np.zeros(epochs)
    for epoch in range(epochs):
        state = env.reset()
        action = ql.querysetstate(state)
        for i in range(max_iters):
            state, reward, done, info = env.step(action)
            action = ql.query(state, reward)
            epoch_rewards[epoch] += reward
            if done or i == max_iters - 1:
                epoch_steps[epoch] = i + 1
                break
    end = time.time()
    elapsed = end - start
    policy = np.argmax(ql.Q, axis=1)
    values = np.max(ql.Q, axis=1)
    return ql, {
        "time": elapsed,
        "epoch_steps": epoch_steps,
        "total_steps": epoch_steps.sum(),
        "epoch_rewards": epoch_rewards,
        "policy": tuple(policy),
        "V": tuple(values)
    }

