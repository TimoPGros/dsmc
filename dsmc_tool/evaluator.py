import gymnasium as gym
from gymnasium import Env as GymEnv
import pgtg
from typing import Dict, Any
from dsmc_tool.eval_results import eval_results
from dsmc_tool.property import Property, ReturnProperty
import dsmc_tool.statistics as stats

# Main evaluator class
class Evaluator:

    # initial_episodes: number of episodes to run before initially (should be considerably larger than subsequent_episodes)
    # subsequent_episodes: number of episodes to run in each iteration after the initial episodes
    def __init__(self, env: GymEnv = gym.make("pgtg-v3"), initial_episodes: int = 100, subsequent_episodes: int = 50):
        self.env = env
        self.initial_episodes = initial_episodes
        self.subsequent_episodes = subsequent_episodes
        self.properties = {}
        self.made_episodes = 0

    # register a new evaluation property
    def register_property(self, property: Property = ReturnProperty(), json_filename: str = None):
        self.properties[property.name] = property
        if json_filename != None:
            property.json_filename = json_filename

    # run the policy for a specified number of episodes
    def __run_policy(self, agent: Any = None, num_episodes: int = 50, results_per_property: Dict[str, eval_results] = None, act_function = None, save_interim_results: bool = False, output_full_results_list: bool = False, initial: bool =False, interim_interval: int = None, truncation_steps: int = None):
        act_function = act_function or agent.get_action           
        if not callable(act_function):
            raise ValueError("act_function should be a function.")        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            trajectory = []
            terminated = False
            truncated = False
            counter = 0
            while not (terminated or truncated):
                counter += 1
                output = act_function(state)
                if isinstance(output, tuple):
                    action = output[0]
                else:
                    action = output
                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    action = int(action)
                else:
                    action = float(action) if isinstance(action, (int, float)) else action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state
                if truncation_steps != None:
                    if counter >= truncation_steps:
                        truncated = True

            # store new results in EvaluationResults object
            self.made_episodes += 1
            for property in self.properties.values():
                results_per_property[property.name].extend(property.check(trajectory))
                results_per_property[property.name].total_episodes += 1
            if save_interim_results:
                if interim_interval == None:
                    raise ValueError("interim_interval must be specified when save_interim_results is True.")
                if self.made_episodes % interim_interval == 0:
                    if initial:
                        for property in self.properties.values():
                            results_per_property[property.name].save_data_interim(filename = property.json_filename, initial = True, output_full_results_list = output_full_results_list)
                        initial = False
                    else:    
                        for property in self.properties.values():
                            results_per_property[property.name].save_data_interim(filename = property.json_filename, output_full_results_list = output_full_results_list)
    
    # evaluate the agent
    #act_function: specifies the function to be used to get the action in your agent implementation (default is agent.get_action, action has to be in the first position of the output)
    #save_interim_results: if True, the results are saved every few episodes
    #interim_interval: specifies the number of episodes after which the results are saved when save_interim_results is True (default is the number of subsequent_episodes)
    #output_full_results_list: if True, the full list of results is saved in the json file
    #relative_epsilon: if True, epsilon is scaled by the mean of the property
    #truncation_steps: number of steps after which the episode is truncated
    def eval(self, agent, epsilon: float = 0.1, kappa: float = 0.05, act_function = None, save_interim_results: bool = False, interim_interval: int = None, output_full_results_list: bool = False, relative_epsilon: bool = False, truncation_steps: int = None):
        # initialize EvaluationResults object for each class and whether the property converged
        if self.properties == {}: 
            raise ValueError("No properties registered. Use register_property to register properties.")
        if interim_interval == None:
            interim_interval = self.subsequent_episodes
        results_per_property = {}
        converged_per_property = {}
        for property in self.properties.values():
            results_per_property[property.name] = eval_results(property=property)
            converged_per_property[property.name] = False

        # run initial episodes
        self.made_episodes = 0       
        self.__run_policy(agent, self.initial_episodes, results_per_property, act_function, save_interim_results = save_interim_results, output_full_results_list=output_full_results_list, initial = True, interim_interval = interim_interval, truncation_steps = truncation_steps)
        # compute the CH bound
        ch_bound = stats.CH(epsilon, kappa)
        # run the policy until all properties have converged
        while True:
            # run the policy for the specified number of episodes
            self.__run_policy(agent, self.subsequent_episodes, results_per_property, act_function, save_interim_results = save_interim_results, output_full_results_list=output_full_results_list, interim_interval = interim_interval, truncation_steps = truncation_steps)
           
            # compute for each property the APMC bound and the confidence interval length
            for property in self.properties.values():
                property_results = results_per_property[property.name]

                apmc_bound = stats.APMC(property_results.get_variance(), epsilon, kappa)
                confidence_interval_length = stats.construct_confidence_interval_length(property_results, kappa)

                if relative_epsilon:
                    val = property_results.get_mean()
                    scaled_epsilon = abs(epsilon * val)
                else:
                    scaled_epsilon = epsilon
                # check if the property has converged, property can also become non-converged again!!!
                if self.made_episodes > ch_bound or self.made_episodes > apmc_bound or confidence_interval_length < 2 * scaled_epsilon:
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