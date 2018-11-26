import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

