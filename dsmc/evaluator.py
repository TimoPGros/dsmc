from dsmc.eval_results import eval_results
import gymnasium as gym
from gymnasium import Env as GymEnv
from dsmc.property import Property, ReturnProperty
import dsmc.statistics as stats

from typing import Dict

class Evaluator:

    def __init__(self, env: GymEnv = gym.make("CartPole-v1"), gamma: float = 0.99, initial_episodes: int = 100, evaluation_episodes: int = 50):
        self.env = env
        self.gamma = gamma
        self.initial_episodes = initial_episodes
        self.evaluation_episodes = evaluation_episodes

        self.properties = {}

    # register a new evaluation property
    def register_property(self, property: Property = ReturnProperty()):
        self.properties[property.name] = property

    def __run_policy(self, agent, num_episodes: int, results_per_property: Dict[str, eval_results], act_function):
        if act_function is None:
            act_function = agent.predict
            
        if not callable(act_function):
            raise ValueError("act_function should be a function.")
        
        for _ in range(num_episodes):
            state = self.env.reset()
            trajectory = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = act_function(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            # store new results in EvaluationResults object
            for property in self.properties.values():
                results_per_property[property.name].extend(property.check(trajectory))

    def eval(self, agent, epsilon: float = 0.1, kappa: float = 0.05, act_function = None):
        # initialize EvaluationResults object for each class and whether the property converged
        results_per_property = {}
        converged_per_property = {}
        for property in self.properties.values():
            results_per_property[property.name] = eval_results(property=property)
            converged_per_property[property.name] = False

        # run initial episodes
        
        self.__run_policy(agent, self.initial_episodes, results_per_property, act_function)
        made_episodes = self.initial_episodes
        for property in self.properties.values():
                results_per_property[property.name].total_episodes = self.initial_episodes

        # compute the CH bound
        ch_bound = stats.CH(kappa, epsilon)
        # run the policy until all properties have converged
        while True:
            # run the policy for the specified number of episodes
            self.__run_policy(agent, self.evaluation_episodes, results_per_property, act_function)
            made_episodes += self.evaluation_episodes
            for property in self.properties.values():
                results_per_property[property.name].total_episodes += self.evaluation_episodes

            # compute for each property the APMC bound and the confidence interval length
            for property in self.properties.values():
                property_results = results_per_property[property.name]

                apmc_bound = stats.APMC(property_results.get_variance(), kappa, epsilon)
                confidence_interval_length = stats.construct_confidence_interval_length(property_results, kappa)

                # check if the property has converged, property can also become non-converged again!!!
                if made_episodes > ch_bound or made_episodes > apmc_bound or confidence_interval_length < 2 * epsilon:
                    converged_per_property[property.name] = True
                else:
                    converged_per_property[property.name] = False

            if all(converged_per_property.values()):
                break

        return results_per_property
