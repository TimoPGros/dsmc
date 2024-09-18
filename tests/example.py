import gymnasium as gym
from dsmc.evaluator import Evaluator

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, _):
        return self.env.action_space.sample()

    def learn(self, total_timesteps):
        for _ in range(total_timesteps):
            state = self.env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = self.predict(state)
                _, _, terminated, truncated, _ = self.env.step(action)

env = gym.make("CartPole-v1")

model = RandomAgent(env)
model.learn(total_timesteps=1000)

evaluator = Evaluator(env=env)
evaluator.register_property()
results = evaluator.eval(model, epsilon=0.1, kappa=0.05)

print(results["return"])
print(results["return"].get_all())
print(results["return"].get_mean())
print(results["return"].get_variance())
print(results["return"].get_std())
print(results["return"].get_confidence_interval())