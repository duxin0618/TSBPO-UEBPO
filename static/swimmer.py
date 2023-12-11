import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done

    @staticmethod
    def reward_fn(obs, act, next_obs):
        reward_ctrl = -0.0001 * np.sum(np.square(act), axis=1)
        reward_run = obs[:, 3]
        reward = reward_run + reward_ctrl
        return reward