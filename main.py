import os
import time
from functools import cached_property
from scipy.stats import norm
import numpy as np
import pytesseract
from gym import Env
from gym.spaces import Box
from mss import mss
from pyautogui import press, click, keyUp, keyDown
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)


class LanderEnv(Env):
    def __init__(self):
        super().__init__()
        # key: dytpe = int, max = 4
        # time: dtype = float, max = 3
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=float)
        self.observation_space = Box(low=-float("inf"), high=float("inf"), shape=(3,))
        self.screen_capture = mss()
        self.game_over_area = {'top': 256, 'left': 261, 'width': 600, 'height': 120}
        self.driver.get("<URL>")
        self.driver.maximize_window()

    @cached_property
    def driver(self):
        return webdriver.Chrome(options=chrome_options)

    def step(self, action):
        action_map = {
            1: 'left',
            2: 'up',
            3: 'right',
            4: ['left', 'right'],
            5: ['up', 'left'],
            6: ['up', 'right'],
            7: 'noop'
        }
        [action_key, time_period] = action
        action_key = int(self.action_to_key(action_key, len(action_map))) + 1
        time_period = self.value_to_time(time_period)
        if action_key < len(action_map):
            key = action_map[action_key]
            if not isinstance(key, list):
                keyDown(action_map[action_key])
                time.sleep(time_period)
                keyUp(action_map[action_key])
            else:
                keyDown(key[0])
                keyDown(key[1])
                time.sleep(time_period)
                keyUp(key[0])
                keyUp(key[1])
        elif action_key == 5:
            time.sleep(time_period)

        obs = self.get_observation()
        done, is_success = self.is_done(obs)
        # obs = self.get_observation()
        reward = self.get_reward(obs, done, is_success)
        info = {}
        return obs, reward, done, info

    def action_to_key(self, action, num_actions):
        bin_boundaries = norm.ppf(np.linspace(0, 1, num_actions + 1))
        return np.searchsorted(bin_boundaries, action) - 1

    def value_to_time(self, value):
        probability = norm.cdf(value)
        return 3 * probability


    def get_reward(self, obs, done, is_sucess):
        '''
        speed: reward: 11 - speed
        angle: reward: -(abs(angle) - 10)
        '''
        if done and is_sucess:
            return 100
        elif done and not is_sucess:
            return -500

        [speed, angle, height] = obs
        speed_reward = 11 - speed
        angle_reward = -(abs(angle) - 10)

        reward = (speed_reward + angle_reward) / 3
        return reward

    def reset(self):
        self.driver.refresh()
        click(672, 307)
        press("space")
        return self.get_observation()

    def get_observation(self):
        stats = self.driver.find_element(By.CSS_SELECTOR, value="#stats span").text.split(",")
        stats = np.array([float(x.strip()) for x in stats])
        return stats

    def close(self):
        pass

    def is_done(self, obs=None):
        height = obs[-1]
        if height > 210:
            return True, False  # done and Failure

        status = np.array(self.screen_capture.grab(self.game_over_area))[:, :, :3].astype(np.uint8)
        res = pytesseract.image_to_string(status).split("\n")[0]
        if "CRASH" in res:
            return True, False  # done and failure
        elif "PERFECT" in res:
            return True, True  # done and sucess

        return False, False  # not done and not sucess


def try_env(env, model=None):
    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if model is None:
                obs, reward, done, info = env.step(env.action_space.sample())
            else:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f'Total Reward for episode {episode} is {total_reward}')


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True


CHECKPOINT_DIR = './ehmorris_lander/'
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)


def train(model):
    model.learn(total_timesteps=50000, callback=callback, log_interval=100)


def load(path, env):
    model = PPO.load(path, env=env)
    return model


if __name__ == "__main__":
    env = LanderEnv()
    vec_env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', vec_env, verbose=1)
    train(model)
    print("Done")

