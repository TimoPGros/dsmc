import numpy as np

# base class for evaluation metrics
class Metric:
        def __init__(self, name):
            self.name = name
            pass

        # we assume a trajectory is a list of tuples (state, action, reward)
        def check(self, trajectory: list) -> float:
            pass

# metric that checks if the goal has been reached
class GoalReachingProbabilityMetric(Metric):
    def __init__(self, name, goal_reward):
        super().__init__(name)
        self.goal_reward = goal_reward

    def check(self, trajectory):
        if trajectory[-1][2] == self.goal_reward:
            return 1.0
        else:
            return 0.0

# metric that calculates the return of a trajectory
class ReturnMetric(Metric):
    def __init__(self, name, gamma):
        super().__init__(name)
        self.gamma = gamma

    def check(self, trajectory):
        ret = 0
        for t in range(len(trajectory)):
            ret += trajectory[t][2] * np.power(self.gamma, t)
        return ret

class Evaluator():

    def __init__(self, env: GymEnv, gamma: float, initial_episodes: int, episodes_per_run: int, ):
        self.env = env
        self.gamma = gamma
        self.initial_runs = initial_episodes
        self.episodes_per_run = episodes_per_run
        self.metrics = {}

    # register a new evaluation metric
    def register_metric(self, metric: Metric):
        self.metrics[metric.name] = metric

    def eval(self, agent, epsilon, kappa):


    def run_policy(self, n_episodes, l_episodes=100):
        for i in range(1, n_episodes + 1):
            obs, _ = self.env.reset()

            ret = 0
            for t in range(l_episodes):
                action, _states = self.act_function(obs)
                obs, r, done, truncated, _ = self.env.step(int(action))

                ret += r * np.power(self.gamma, t)

                if (done or truncated):
                    break

            self._update(ret)


    # grps_eps: The Accuracy to be achieved for the ratio between wins and runs in general
    # return_eps: The Accuracy to be achieved for the expected reward
    # kappa: How sure we want to be we achieved accuracy the eps

    # TODO: evaluate arbitrary python functions based on trajectory
    # TODO: correction term
    # TODO: make grp optional

    def eval(self, grps_eps, return_eps, kappa, l_episodes=100,
             initial_runs=50):  # TODO: support absolute and relative epsilon
        start = time.time()
        ch_bound = st.CH(kappa, grps_eps)  # chernoff-hoeffding bound, number of required samples

        made_runs = 0
        while True:
            logging.debug("starting the next batch of simulations, performed totally %d" % made_runs)
            self._simulate_runs(initial_runs, l_episodes=l_episodes)
            made_runs += initial_runs
            grps_variance = st.binomial_variance(self.loses + self.tos, self.wins, 0,
                                                 1)  # variance of the binomial distribution for the ratio between wins and runs in general
            # helps determine the accuracy and confidence of the estimated proportion of wins
            apmc_bound = st.APMC(grps_variance, kappa,
                                 grps_eps)  # required sample size to ensure that estimated mean is within accuracy grps_eps with confidence kappa
            # treat binomial as normal distribution (CLT)

            interval_length_r = st.construct_confidence_interval_length(self.returns,
                                                                        kappa)  # length of a confidence interval for the mean of the return
            interval_length_grp = st.construct_binomial_confidence_interval_length(self.loses + self.tos, self.wins,
                                                                                   kappa=kappa)  # length of a confidence interval for the ratio

            return_criterion = interval_length_r < 2 * return_eps  # one side of the interval has to be smaller than our accuracy
            grps_criterion = (
                    (made_runs > apmc_bound)  # these bounds already tell us the maximum number of necessary episodes
                    or (made_runs > ch_bound)
                    or (interval_length_grp < 2 * grps_eps)  # see above
            )
            print(self.results())  # TODO: Improve the Output (save to File)
            if (return_criterion and grps_criterion):
                break

        end = time.time()
        print("Finished evaluation of agent in %.2f sec using %d eps" % ((end - start), self._sim_len()))

    def results(self, exhaustive=False):
        if exhaustive:
            return self.returns[-1], self.returns[-2], self.returns[-3], self.returns[-4]
        else:
            return np.mean(self.returns), self.wins / (self.wins + self.loses + self.tos)
