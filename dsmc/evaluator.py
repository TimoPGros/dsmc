import numpy as np


def CH(kappa, eps):
    x = 1 / np.power(eps, 2)
    y = np.log(2 / (kappa))
    res = x * y

    return int(np.floor(res))

def APMC(s2, kappa, eps):
    z = st.norm.ppf(1 - kappa / 2)

    return np.ceil(4 * z * s2 / np.power(eps, 2))

def get_variance(results, epsilon, kappa):
    pass

def construct_confidence_interval_length(results, epsilon, kappa):
    pass


# base class for evaluation properties
class Property:
        def __init__(self, name):
            self.name = name
            pass

        # we assume a trajectory is a list of tuples (state, action, reward)
        def check(self, trajectory: list) -> float:
            pass

# metric that checks if the goal has been reached
class GoalReachingProbabilityProperty(Property):
    def __init__(self, name, goal_reward):
        super().__init__(name)
        self.goal_reward = goal_reward

    def check(self, trajectory):
        if trajectory[-1][2] == self.goal_reward:
            return 1.0
        else:
            return 0.0

# metric that calculates the return of a trajectory
class ReturnProperty(Property):
    def __init__(self, name, gamma):
        super().__init__(name)
        self.gamma = gamma

    def check(self, trajectory):
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret

class Evaluator:

    def __init__(self, env: GymEnv, gamma: float, initial_episodes: int, episodes_per_run: int, ):
        self.env = env
        self.gamma = gamma
        self.initial_episodes = initial_episodes
        self.episodes_per_run = episodes_per_run
        self.properties = {}

    # register a new evaluation property
    def register_metric(self, property: Property):
        self.properties[property.name] = property

    def run_policy(self, agent, num_episodes, results_per_property):
        for _ in range(num_episodes):
            state = self.env.reset()
            trajectory = []
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            for property in self.properties.values():
                results_per_property[property.name].append(property.check(trajectory))

    def eval(self, agent, epsilon, kappa):
        # initialize EvaluationResults object for each class and whether the property converged
        results_per_property = {}
        converged_per_property = {}
        for property in self.properties.values():
            results_per_property[property.name] = EvaluationResults(property=property.name)
            converged_per_property[property.name] = False

        # run initial episodes - one run, such that the first run of the while loop checks convergence
        self.run_policy(agent, self.initial_episodes - self.episodes_per_run, results_per_property)
        made_episodes = self.initial_episodes - self.episodes_per_run

        # compute the CH bound
        ch_bound = CH(kappa, epsilon)
        # run the policy until all properties have converged
        while True:
            # run the policy for the specified number of episodes
            self.run_policy(agent, self.episodes_per_run, results_per_property)
            made_episodes += self.episodes_per_run

            # compute for each property the apmc bound and the confidence interval length
            for property in self.properties.values():
                variance = get_variance(results_per_property[property.name], epsilon, kappa)
                apmc_bound = APMC(get_variance(variance, kappa, epsilon))
                confidence_interval_length = construct_confidence_interval_length(results_per_property[property.name], epsilon, kappa)

                # check if the property has converged
                if made_episodes > ch_bound or made_episodes > apmc_bound or confidence_interval_length < 2 * epsilon:
                    converged_per_property[property.name] = True
            
            if all(converged_per_property.values()):
                break

