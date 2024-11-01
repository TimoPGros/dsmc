import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pgtg
from stable_baselines3 import DQN, DDPG
from dsmc_tool.evaluator import Evaluator
import dsmc_tool.property as prop

#env = gym.make("pgtg-v3")
env = gym.make("MountainCarContinuous-v0")
env = FlattenObservation(env)
agent = None
if isinstance(env.action_space, gym.spaces.Discrete):
    agent = DQN("MlpPolicy", env, verbose=1)
else:
    agent = DDPG("MlpPolicy", env,verbose=1)
agent.learn(total_timesteps=1000, log_interval=1)

evaluator = Evaluator(env=env, initial_episodes=100, subsequent_episodes=50)
property = prop.ReturnProperty(gamma=1)
evaluator.register_property(property)
results = evaluator.eval(agent, epsilon=0.05, kappa=0.025, act_function=agent.predict, save_interim_results=True, relative_epsilon=True)