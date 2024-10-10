import json
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import t
import dsmc.property as prop

# Class to store the results of the evaluation of a property
class eval_results:
    
    def __init__(self, property: prop.Property = prop.ReturnProperty()):
        self.__result_dict = np.array([])
        self.property = property
        self.total_episodes = 0
        self.var = None
        self.mean = None
        self.std = None
        self.anything_written = False
    
    # Returns the whole array of results
    def get_all(self):
        return self.__result_dict
    
    # Returns the mean of the results     
    def get_mean(self):
        mean = np.mean(self.__result_dict) 
        self.mean = mean
        return mean      
    
    # Returns the variance of the results
    def get_variance(self):
        if len(self.__result_dict) == (0 or 1):
            return 0.0
        if self.property.binomial:
            num1 = self.__result_dict.sum()
            num0 = self.total_episodes - num1
            n = num0 + num1
            x = num0 / n * 0.0 + num1 / n * 1.0
            if (n==1):
                var = 0.0
            else:
                var = num0 / (n - 1) * np.power((0.0 - x), 2) + num1 / (n - 1) * np.power((1.0 - x), 2)
            self.var = var 
            return var   
        else:
            var = np.var(self.__result_dict, ddof=1) 
            self.var = var
            return var
               
    # Returns the standard deviation of the results
    def get_std(self):
        var = None
        if self.var is None:
            var = self.get_variance()
        else:
            var = self.var
        std = np.sqrt(var)
        self.std = std     
        return std
    
    # Returns the confidence interval of the results, according to kappa  
    def get_confidence_interval(self, kappa: float = 0.05):
        if self.property.binomial:
            num1 = self.__result_dict.sum()
            num0 = self.total_episodes - num1
            n = num0 + num1
            mean = num0 / n * 0.0 + num1 / n * 1.0
            std = None
            if self.std is None:
                std = self.get_std()
            else:
                std = self.std
            t_stat = norm.ppf((kappa / 2, 1 - kappa / 2))[-1]
            interval = [
                mean - t_stat * std / np.sqrt(n),
                mean + t_stat * std / np.sqrt(n),
            ]

            return interval
        else:
            mean = None
            std = None
            if self.mean is None:
                mean = self.get_mean()
            else:
                mean = self.mean
            if self.std is None:
                std = self.get_std()
            else:
                std = self.std
            n = len(self.__result_dict)
            if (n == 1):
                return [mean, mean]
            else:
                t_stat = t(df=n - 1).ppf((kappa / 2, 1 - kappa / 2))[-1]
                interval = [
                    mean - t_stat * std / np.sqrt(n),
                    mean + t_stat * std / np.sqrt(n),
                ]

            return interval
    
    # Adds a new result to the array
    def extend(self, new_result: float = 0.0):
            self.__result_dict = np.append(self.__result_dict, np.array([new_result]))          
    
    # Only saves information about the property once evaluation is finished
    # output_full_results_list: if True, the full list of results is saved in the json file
    def save_data_end(self, filename: str = None, output_full_results_list: bool = False):
        if not filename.endswith(".json"):
            filename += ".json"
        data = {}
        data['property'] = self.property.name
        if output_full_results_list:
            data['full_results_list'] = self.get_all().tolist()
        data['mean'] = self.get_mean()
        data['variance'] = self.get_variance()
        data['std'] = self.get_std()
        data['confidence_interval'] = self.get_confidence_interval()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        self.anything_written = True
        print(f"Data saved to {filename}")
    
    # Saves information about the property every few episodes (number can be specified when running the evaluation)
    # initial: Set to True when called for the first time in the evaluation
    # final: Set to True when called for the last time in the evaluation
    # output_full_results_list: if True, the full list of results is saved in the json file
    def save_data_interim(self, filename: str = None, initial: bool = False, final: bool = False, output_full_results_list: bool = False):
        if not filename.endswith(".json"):
            filename += ".json"
        if initial:
            data = {}
            data['property'] = self.property.name
            if output_full_results_list:
                data[str(self.total_episodes)] = {
                    'full_results_list': self.get_all().tolist(),
                    'mean': self.get_mean(),
                    'variance': self.get_variance(),
                    'std': self.get_std(),
                    'confidence_interval': self.get_confidence_interval()
                }
            else:
                data[str(self.total_episodes)] = {
                    'mean': self.get_mean(),
                    'variance': self.get_variance(),
                    'std': self.get_std(),
                    'confidence_interval': self.get_confidence_interval()
                }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            self.anything_written = True
        else:
            if os.path.exists(filename) and self.anything_written:
                with open(filename, 'r') as f:
                    data = json.load(f)
            else:
                with open(filename, 'w') as f:
                    pass
                data = {}
                data['property'] = self.property.name
            name = None
            if final:
                name = 'final'
            else:
                name = str(self.total_episodes) 
            if output_full_results_list:
                data[name] = {
                    'full_results_list': self.get_all().tolist(),
                    'mean': self.get_mean(),
                    'variance': self.get_variance(),
                    'std': self.get_std(),
                    'confidence_interval': self.get_confidence_interval()
                }
            else:   
                data[name] = {
                    'mean': self.get_mean(),
                    'variance': self.get_variance(),
                    'std': self.get_std(),
                    'confidence_interval': self.get_confidence_interval()
                }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            self.anything_written = True
        if final:
            print(f"Data saved to {filename}")
          