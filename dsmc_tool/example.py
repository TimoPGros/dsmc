import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pgtg
from stable_baselines3 import DQN
from dsmc_tool.evaluator import Evaluator
import dsmc_tool.property as prop

env = gym.make("pgtg-v3")
env = FlattenObservation(env)

model = DQN("MlpPolicy", env,verbose=1)
model.learn(total_timesteps=1000, log_interval=100)

evaluator = Evaluator(env=env, initial_episodes=100, evaluation_episodes=50)
property = prop.ReturnProperty(name="relative_return")
evaluator.register_property(property)
results = evaluator.eval(model, epsilon=0.05, kappa=0.025, save_interim_results=True, relative_epsilon=True)