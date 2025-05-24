import gymnasium as gym


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))

    def reset(self):
        return self.observation_space.sample(), {}
