import gymnasium as gym
from dsmc.evaluator import Evaluator
import dsmc.property as prop

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
grp_prop = prop.GoalReachingProbabilityProperty()
evaluator.register_property(grp_prop)
results = evaluator.eval(model, epsilon=0.1, kappa=0.05, save_interim_results=True)