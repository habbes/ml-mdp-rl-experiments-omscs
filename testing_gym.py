import gym
import mdptoolbox as mtools
from mdptoolbox.mdp import PolicyIteration, ValueIteration
import registry
import lib

registry.register_envs()

env = gym.make('SmallLake-v0')

T, R = lib.get_mdptbx_env_from_gym_env(env)
pi = PolicyIteration(T, R, 0.9)
pi.run()



for i_episodes in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("observation", observation)
        action = pi.policy[observation]
        observation, reward, done, info = env.step(action)
        print("action", action, "reward", reward)
        if done:
            print("Finished after {} steps".format(t + 1))
            break