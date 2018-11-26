from gym.envs.registration import registry, register, make, spec

def register_envs():
    register(
        id='SmallLake-v0',
        entry_point='custom_lake:CustomLakeEnv',
        kwargs={'map_name' : '4x4'},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )

    register(
        id='SmallLakeTough-v0',
        entry_point='custom_lake:CustomLakeEnv',
        kwargs={'map_name' : '4x4', 'hole_reward': -1.0},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )

    register(
        id='LargeLake-v0',
        entry_point='custom_lake:CustomLakeEnv',
        kwargs={'map_name' : '8x8', 'uniform_action_prob': True},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )

    register(
        id='LargeLakeTough-v0',
        entry_point='custom_lake:CustomLakeEnv',
        kwargs={'map_name' : '8x8', 'uniform_action_prob': False, 'hole_reward': -1.0},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
    