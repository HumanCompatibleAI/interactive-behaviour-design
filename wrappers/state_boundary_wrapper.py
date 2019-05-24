from gym import Wrapper

# In a separate file to avoid circular import caused by SubprocVecEnv and VecSaveSegments


class StateBoundaryWrapper(Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)