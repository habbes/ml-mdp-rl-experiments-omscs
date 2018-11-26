import gym
import mdptoolbox as mtools
from mdptoolbox.mdp import PolicyIteration, ValueIteration
import registry
import lib

registry.register_envs()

env = gym.make('SmallLakeTough-v0')

T, R = lib.get_mdptbx_env_from_gym_env(env)
pi = ValueIteration(T, R, 0.9)
pi.run()


sum_rewards = 0
sum_steps = 0
for i_episodes in range(1000):
    observation = env.reset()
    for t in range(100):
        # env.render()
        # print("observation", observation)
        action = pi.policy[observation]
        observation, reward, done, info = env.step(action)
        # print("action", action, "reward", reward)
        sum_steps += 1
        sum_rewards += reward
        if t == 99:
            print("NOT DONE AFTER 100", reward)
        if done:
            # print("Finished after {} steps".format(t + 1))
            if reward < 1.0:
                print("FINISHED WITH", reward, observation)
            break

avg_reward = sum_rewards / 1000.0
avg_steps = sum_steps / 1000.0
print("avg reward", avg_reward, "avg steps", avg_steps)