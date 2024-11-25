# DSMC-Tool

DSMC-Tool is a package for Deep Statistical Model Checking (DSMC) of Deep Reinforcement Learning agents. This application allows users to evaluate RL agents based on a wide array of properties, making it an essential tool for ensuring robust agent performance. The package is developed to be fully compatible with Gymnasium environments.

## Table of Contents
- [DSMC](#dsmc)
- [Evaluator Initialization](#evaluator-initialization)
- [Properties](#properties)
- [Evaluation](#evaluation)
- [Example](#example)

## DSMC

In a nutshell, DSMC (Deep Statistical Model Checking) leverages a DRL (Deep Reinforcement Learning) agent to run a series of episodes in a given environment and statistically estimate a property of interest. The process involves creating a confidence interval around the estimate, based on the provided parameter `kappa`, and comparing it against the specified accuracy parameter `epsilon`. This estimation process operates iteratively: after generating a certain number of episodes, the algorithm checks whether the estimate satisfies the desired accuracy. If not, additional episodes are generated, and the process repeats. The loop ensures termination as the confidence interval becomes tighter with more episodes, eventually meeting the specified criteria.

## Evaluator Initialisation

The first step in using DSMC-Tool is initialising an `Evaluator` object from `dsmc_tool.evaluator`. The following constructor parameters can be set:

|Parameter           |Type    |Default Value       |Description                                                         |
|--------------------|--------|--------------------|--------------------------------------------------------------------|
|env                 |Env     |None                |Your RL environment, defined as a Gymnasium environment             |
|initial_episodes    |int     |100                 |The number of episodes generated in the initial run of evaluation   |
|subsequent_episodes |int     |50                  |The number of episodes generated in every run after the initial run |

Note that the number of initial episodes should be chosen relatively high to avoid premature termination.

## Properties 

After initialisation, one or more properties have to be created and registered for evaluation. Properties are implementations of the abstract class `Property`, which is provided in `dsmc_tool.property`:

```python
class Property:
    def __init__(self, name: str):
        self.name = name
        self.json_filename = name + ".json"
        pass

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        pass
```
By default, the name of the JSON file containing the output is identical to the property name, but this can be changed in your implementation or in the `register` function. The `check` function is used to derive a result for the given episode, provided in the form of `trajectory`. This is a List of tuples with each tuple describing a single time step by containing the current observation, action, and reward. For example, this is the implementation of a Property calculating the return:

```python
class ReturnProperty(Property):
    def __init__(self, name: str = "return", gamma: float = 0.99):
        super().__init__(name)
        self.gamma = gamma
        self.binomial = False

    def check(self, trajectory: List[Tuple[Any, Any, Any]]) -> float:
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret
```

Additionally to the possibility of creating custom properties, there is a library of pre-implemented properties:

| **Name**                          | **Additional Inputs**       | **Description**                                                                                                                                             |
|------------------------------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ActionDiversityProperty           | Number of actions           | Calculates the ratio of unique actions taken in the episode to the total number of actions                                                                  |
| ActionEntropyProperty             | Number of actions           | Calculates action entropy, a measure of how much randomness is involved in the action decisions                                                             |
| ActionTakenProperty               | Action                      | Tests whether the given action was applied in the episode                                                                                                   |
| ActionThresholdProperty           | Action, Threshold           | Tests whether the given action was applied a number of times higher than or equal to the threshold                                                          |
| ActionVarietyProperty             | Threshold                   | Tests whether a number of actions higher than or equal to the threshold was applied                                                                         |
| ConsecutiveSameActionProperty     | Action, Threshold           | Tests whether the given action was applied in a consecutive number of time steps higher than or equal to the threshold                                      |
| EarlyTerminationProperty          | Step maximum                | Tests whether the episode terminated within a number of time steps lower than or equal to the step maximum                                                  |
| EpisodeLengthProperty             | None                        | Calculates the length of the episode                                                                                                                        |
| GoalBeforeStepLimitProperty       | Goal reward, Step limit     | Tests whether the goal, signified by a unique goal reward, was reached within a number of time steps lower than or equal to the step limit                  |
| GoalReachingProbabilityProperty   | Goal reward                 | Tests whether the goal, signified by a unique goal reward, was reached                                                                                      |
| NormalizedReturnProperty          | Gamma                       | Calculates the return discounted with discount factor gamma and normalized by the episode length                                                            |
| PathEfficiencyProperty            | Path                        | Calculates the percentage of taken actions that correspond to the actions taken in the given path                                                           |
| PathLengthEfficiencyProperty      | Path length                 | Calculates the ratio of the length of the episode to the given path length                                                                                  |
| ReturnProperty                    | Gamma                       | Calculates the return discounted with discount factor gamma                                                                                                 |
| ReturnThresholdProperty           | Gamma, Threshold            | Tests whether the return discounted with discount factor gamma is higher than or equal to the threshold                                                     |
| RewardToLengthRatioProperty       | Gamma, Threshold            | Calculates the ratio of the sum of all rewards to the length of the episode                                                                                 |
| RewardVarianceProperty            | Gamma, Threshold            | Calculates the variance in the rewards accumulated in the episode                                                                                           |
| StateCoverageProperty             | Number of states            | Calculates the ratio of unique states visited in the episode to the total number of states                                                                  |
| StateTransitionSmoothnessProperty | Number of states            | Calculates the ratio of unique states visited in the episode to the total number of states (assumes states are represented as vectors)                      |
| StateVisitProperty                | State                       | Tests whether the given state was visited in the episode                                                                                                    |

A property can be registered for evaluation using the evaluator's `register` functions. Additionally to the property object, you can provide a custom name for the output JSON file here.

## Evaluation

Once at least one property has been registered, the `eval` function can be called. This function provides a lot of configuration via the input variables, presented here: 

|Parameter                |Type       |Default Value       |Description                                                         |
|-------------------------|-----------|--------------------|--------------------------------------------------------------------|
|agent                    |None given |None                |Your DRL agent implementation                                       |
|epsilon                  |float      |0.1                 |The accuracy parameter (see section DSMC)                           |
|kappa                    |float      |0.05                |The confidence parameter (see section DSMC)                         |
|exploration_rate         |float      |None                |Probability of choosing a random action during evaluation            |
|act_function             |None given |None                |The function your agent implementation uses to decide on an action  |
|save_interim_results     |bool       |False               |Whether interim results should be saved in the output files         |
|interim_interval         |int        |None                |How many episodes should be between the interim results             |
|output_full_results_list |bool       |False               |Whether a list of all results should be saved in the output files   |
|relative_epsilon         |bool       |False               |Whether epsilon should be used relative to the estimate             |
|truncation_steps         |int        |None                |After how many steps evaluation episodes should be truncated        |

If `save_interim_results` is False, the results will only be saved one time, once the evaluation has ended for all properties. Otherwise, the results are saved every `interim_interval` episodes. In general, the output files have the JSON format, and hold information about the mean, variance, standard deviation, and confidence interval (according to kappa) in regard to the corresponding property's results. Additionally to these files, the function returns a dictionary `results_per_property`, which holds an `Evaluation_results` object for every evaluated property. These objects allow you to do all calculations from the output files manually.

## Example

Here is an implementation example of all of this combined:

```python
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from dsmc_tool.evaluator import Evaluator
import dsmc_tool.property as prop

env = gym.make("CartPole-v1")
env = FlattenObservation(env)
agent = DQN("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=1000)

evaluator = Evaluator(env=env, initial_episodes=100, subsequent_episodes=50)
property = prop.ReturnProperty()
evaluator.register_property(property)
results = evaluator.eval(agent, epsilon=2, kappa=0.05, act_function=agent.predict, save_interim_results=True)
```