import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from dsmc.evaluator import Evaluator
import dsmc.property as prop
import pgtg
from stable_baselines3 import DQN

env = gym.make("pgtg-v3")
env = FlattenObservation(env)

model = DQN("MlpPolicy", env,verbose=1)
model.learn(total_timesteps=1000, log_interval=100)

evaluator = Evaluator(env=env)
early_property = prop.EarlyTerminationProperty(threshold=5)
evaluator.register_property(early_property)
results = evaluator.eval(model, epsilon=0.1, kappa=0.05, save_interim_results=True)