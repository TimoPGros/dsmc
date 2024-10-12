import gymnasium as gym
from gymnasium import Env as GymEnv
import pgtg
from typing import Dict, Any
from dsmc_tool.eval_results import eval_results
from dsmc_tool.property import Property, ReturnProperty
import dsmc_tool.statistics as stats

# Main evaluator class
class Evaluator:

    # initial_episodes: number of episodes to run before initially
    # evaluation_episodes: number of episodes to run in each iteration after the initial episodes
    def __init__(self, env: GymEnv = gym.make("pgtg-v3"), gamma: float = 0.99, initial_episodes: int = 100, evaluation_episodes: int = 50):
        self.env = env
        self.gamma = gamma
        self.initial_episodes = initial_episodes
        self.evaluation_episodes = evaluation_episodes
        self.properties = {}
        self.made_episodes = 0

    # register a new evaluation property
    def register_property(self, property: Property = ReturnProperty(), json_filename: str = None):
        self.properties[property.name] = property
        if json_filename != None:
            property.json_filename = json_filename

    # run the policy for a specified number of episodes
    def __run_policy(self, agent: Any = None, num_episodes: int = 50, results_per_property: Dict[str, eval_results] = None, act_function = None, save_interim_results: bool = False, output_full_results_list: bool = False, initial: bool =False, interim_interval: int = None):
        act_function = act_function or agent.predict            
        if not callable(act_function):
            raise ValueError("act_function should be a function.")        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            trajectory = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = act_function(state)
                action = int(action)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            # store new results in EvaluationResults object
            self.made_episodes += 1
            for property in self.properties.values():
                results_per_property[property.name].extend(property.check(trajectory))
                results_per_property[property.name].total_episodes += 1
            if save_interim_results:
                if self.made_episodes % interim_interval == 0:
                    if initial:
                        for property in self.properties.values():
                            results_per_property[property.name].save_data_interim(filename = property.json_filename, initial = True, output_full_results_list = output_full_results_list)
                        initial = False
                    else:    
                        for property in self.properties.values():
                            results_per_property[property.name].save_data_interim(filename = property.json_filename, output_full_results_list = output_full_results_list)
    
    # evaluate the agent
    #act_function: specifies the function to be used to get the action in your agent implementation
    #save_interim_results: if True, the results are saved every few episodes (number can be specified when running the evaluation)
    #output_full_results_list: if True, the full list of results is saved in the json file
    def eval(self, agent, epsilon: float = 0.1, kappa: float = 0.05, act_function = None, save_interim_results: bool = False, output_full_results_list: bool = False):
        interim_interval = None
        if save_interim_results:
            interim_interval = int(input("Enter the number of episodes after which interim results should be saved: "))
        # initialize EvaluationResults object for each class and whether the property converged
        results_per_property = {}
        converged_per_property = {}
        for property in self.properties.values():
            results_per_property[property.name] = eval_results(property=property)
            converged_per_property[property.name] = False

        # run initial episodes
        self.made_episodes = 0       
        self.__run_policy(agent, self.initial_episodes, results_per_property, act_function, save_interim_results = save_interim_results, output_full_results_list=output_full_results_list, initial = True, interim_interval = interim_interval)
        # compute the CH bound
        ch_bound = stats.CH(epsilon, kappa)
        # run the policy until all properties have converged
        while True:
            # run the policy for the specified number of episodes
            self.__run_policy(agent, self.evaluation_episodes, results_per_property, act_function, save_interim_results = save_interim_results, output_full_results_list=output_full_results_list, interim_interval = interim_interval)
           
            # compute for each property the APMC bound and the confidence interval length
            for property in self.properties.values():
                property_results = results_per_property[property.name]

                apmc_bound = stats.APMC(property_results.get_variance(), epsilon, kappa)
                confidence_interval_length = stats.construct_confidence_interval_length(property_results, kappa)

                # check if the property has converged, property can also become non-converged again!!!
                if self.made_episodes > ch_bound or self.made_episodes > apmc_bound or confidence_interval_length < 2 * epsilon:
                    converged_per_property[property.name] = True
                else:
                    converged_per_property[property.name] = False

            # if all properties have converged, save the results and break the loop
            if all(converged_per_property.values()):
                if not save_interim_results:
                    for property in self.properties.values():
                        property_results = results_per_property[property.name]
                        property_results.save_data_end(filename = property.json_filename, output_full_results_list = output_full_results_list)
                else:
                    for property in self.properties.values():
                        property_results = results_per_property[property.name]
                        property_results.save_data_interim(filename = property.json_filename, final = True, output_full_results_list = output_full_results_list)
                break

        return results_per_property