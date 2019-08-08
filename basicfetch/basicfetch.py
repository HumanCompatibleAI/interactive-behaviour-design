import os

import numpy as np
from gym.envs import register as gym_register
from gym.envs.robotics import utils
from gym.envs.robotics.robot_env import RobotEnv
from gym.utils import EzPickle
from gym.wrappers import FlattenDictWrapper

from basicfetch.reward_functions import reward_function_dict


class FetchEnvBasic(RobotEnv, EzPickle):
    def __init__(self, delta_control):
        model_path = os.path.join(os.path.dirname(__file__), 'mujoco-py/xmls/fetch/main.xml')
        RobotEnv.__init__(self,
                          model_path=model_path,
                          n_substeps=20,  # copied from FetchEnv
                          n_actions=8,  # 8 actuators
                          initial_qpos=None)
        EzPickle.__init__(self, delta_control)
        self.delta_control = delta_control
        self.reward_func = reward_function_dict['dummy']
        self.random_initial_gripper_position = True

    def get_ctrl_names(self):
        return self.sim.model.actuator_names

    def _env_setup(self, initial_qpos):
        # Position the robot base
        base_initial_qpos = {
            'slide0': 0.405,
            'slide1': 0.48,
            'slide2': 0.0,
        }
        for name, value in base_initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        # Position the object
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = [1.3, 0.75, 0.4]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

    def _reset_sim(self):
        RobotEnv._reset_sim(self)

        # Move end-effector into position

        gripper_rotation = np.array([1., 0., 1., 0.])
        if self.random_initial_gripper_position:
            # Limits of actuation: right on the back edge to just before the front edge
            x = np.random.uniform(1.05, 1.5)
            # A little beyond the left edge to a little beyond the right edge
            y = np.random.uniform(0.3, 1.2)
            # Right on the table to medium-high
            z = np.random.uniform(0.45, 0.8)
            gripper_target = np.array([x, y, z])
        else:
            # Above middle of table
            gripper_target = [1.3419, 0.7491, 0.755]
        self.sim.data.set_mocap_pos('mocap', gripper_target)
        self.sim.data.set_mocap_quat('mocap', gripper_rotation)

        self.sim.model.eq_active[0] = 1  # Enable mocap
        # Not sure what this does, but FetchEnv has it
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        for _ in range(10):
            self.sim.step()
            # The actuators will try to fight against the mocap control.
            # From https://github.com/openai/gym/issues/234, it seems like we can't
            # turn off the actuators (by adjusting e.g. actuator_gainprm).
            # So instead, we just update the actuators to be closer to where the
            # mocap is trying to pull them.
            for n, actuator_name in enumerate(self.sim.model.actuator_names):
                pos = self.sim.data.get_joint_qpos(actuator_name)
                self.sim.data.ctrl[n] = pos

        # Disable mocap and give gripper time to stop bouncing
        self.sim.model.eq_active[0] = 0
        for _ in range(10):
            self.sim.step()

        return True

    def get_actuator_params(self):
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        return ctrlrange, actuation_range, actuation_center

    def normalized_action_to_actuator_ctrl(self, action):
        ctrlrange, actuation_range, actuation_center = self.get_actuator_params()
        ctrl = actuation_center + action * actuation_range
        ctrl = np.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        return ctrl

    def actuator_ctrl_to_normalized_action(self, ctrl):
        _, actuation_range, actuation_center = self.get_actuator_params()
        action = (ctrl - actuation_center) / actuation_range
        action = action[:-1]  # drop the extra gripper finger
        return action

    def _set_action(self, data):
        assert data.shape == (8,)
        if not self.delta_control:
            action = data
        else:
            action_delta = data / 50
            action_delta[-1] *= 50  # move the gripper fingers faster
            action = self.actuator_ctrl_to_normalized_action(self.sim.data.ctrl)
            action += action_delta
            action = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.concatenate([action, [action[-1]]])  # control the fingers together
        ctrl = self.normalized_action_to_actuator_ctrl(action)
        self.sim.data.ctrl[:] = ctrl

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def _sample_goal(self):
        return np.array((0, 0, 0))

    def compute_reward(self, achieved_goal, desired_goal, info):
        quat = self.sim.data.get_body_xquat('gripper_link')
        pos = self.sim.data.get_site_xpos('grip')
        return self.reward_func(quat, pos)

    def _get_obs(self):
        gripper_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        gripper_vel = self.sim.data.get_site_xvelp('grip') * dt
        obs = np.concatenate([gripper_pos, gripper_vel])
        return {
            'observation': obs.copy(),
            'achieved_goal': gripper_pos,
            'desired_goal': self.goal
        }


def make_env(delta=False):
    env = FetchEnvBasic(delta)
    env = FlattenDictWrapper(env, ['observation'])
    return env


def register():
    gym_register(
        id=f'FetchBasic-v0',
        entry_point=make_env,
        max_episode_steps=30
    )
    gym_register(
        id=f'FetchBasicDelta-v0',
        entry_point=lambda: make_env(delta=True),
        max_episode_steps=30
    )
