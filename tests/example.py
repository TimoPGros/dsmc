import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from dsmc.evaluator import Evaluator
import dsmc.property as prop
import pgtg
from stable_baselines3 import DQN

env = gym.make("pgtg-v3")
env = FlattenObservation(env)

model = DQN("MlpPolicy", env,verbose=1)
model.learn(total_timesteps=1000000, log_interval=100)

evaluator = Evaluator(env=env)
return_prop = prop.ReturnProperty()
evaluator.register_property(return_prop)
norm_return_prop = prop.NormalizedReturnProperty()
evaluator.register_property(norm_return_prop)
length_prop = prop.EpisodeLengthProperty()
evaluator.register_property(length_prop)
action_diversity_prop = prop.ActionDiversityProperty()
evaluator.register_property(action_diversity_prop)
state_coverage_prop = prop.StateCoverageProperty()
evaluator.register_property(state_coverage_prop)
results = evaluator.eval(model, epsilon=0.1, kappa=0.01, save_interim_results=True)