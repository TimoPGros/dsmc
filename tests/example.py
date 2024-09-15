import gymnasium as gym
from stable_baselines3 import DQN
from dsmc.evaluator import Evaluator

env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env,verbose=1)
model.learn(total_timesteps=1000)

evaluator = Evaluator(env=env)
evaluator.register_property()
results = evaluator.eval(model, epsilon=0.1, kappa=0.05)
print(results.get_all())
print(results.get_mean())
print(results.get_variance())
print(results.get_std())
print(results.get_confidence_interval(0.05))