import numpy as np
from scipy.stats import norm
from scipy.stats import t

class eval_results:
    
    def __init__(self, property):
        self.__result_dict = np.array([])
        self.property = property
        self.total_episodes = 0
        self.var = None
        self.mean = None
        self.std = None
    
    def get_all(self):
        return self.__result_dict
            
    def get_mean(self):
        mean = np.mean(self.__result_dict) 
        self.mean = mean
        return mean      
    
    def get_variance(self):
        if self.property.binomial:
            num1 = self.__result_dict.sum()
            num0 = self.total_episodes - num1
            n = num0 + num1
            x = num0 / n * 0.0 + num1 / n * 1.0
            var = num0 / (n - 1) * np.power((0.0 - x), 2) + num1 / (n - 1) * np.power((1.0 - x), 2)
            self.var = var 
            return var   
        else:
            var = np.var(self.__result_dict) 
            self.var = var
            return var
               
    
    def get_std(self):
        var = None
        if self.var is None:
            var = self.get_variance()
        else:
            var = self.var
        std = np.sqrt(var)
        self.std = std     
        return std
        
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
            t_stat = t(df=n - 1).ppf((kappa / 2, 1 - kappa / 2))[-1]
            interval = [
                mean - t_stat * std / np.sqrt(n),
                mean + t_stat * std / np.sqrt(n),
            ]

            return interval
           
    def extend(self, new_result: float):
            self.__result_dict = np.append(self.__result_dict, np.array([new_result]))
       
    def save_data(self, filename = 'eval_results.csv'):
        try:
            np.savetxt(filename, self.__result_dict, delimiter=',')
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

            