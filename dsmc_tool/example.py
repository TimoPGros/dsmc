import gymnasium as gym
import pgtg
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from dsmc_tool.evaluator import Evaluator
import dsmc_tool.property as prop

env = gym.make("pgtg-v3")
env = FlattenObservation(env)
agent = DQN("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=1000)

evaluator = Evaluator(env=env, initial_episodes=100, subsequent_episodes=50)
property = prop.ReturnProperty()
evaluator.register_property(property)
results = evaluator.eval(agent, epsilon=2, kappa=0.05, act_function=agent.predict, save_interim_results=True)