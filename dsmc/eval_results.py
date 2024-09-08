import numpy as np

class eval_results:
    
    #TODO: getter for std, confidence interval
    
    def __init__(self, property: str):
        self.__result_dict = np.array([])
        self.property = property
        self.total_episodes = 0
    
    def get_all(self):
        return self.__result_dict
            
    def get_mean(self):
        return np.mean(self.__result_dict)       
    
    def get_variance(self):
        return np.var(self.__result_dict)        
    
    def get_std(self):
        #return np.std(self.__result_dict)       
        pass
        
    def get_confidence_interval(self, kappa: float, epsilon: float):
        #length = len(self.__result_dict)
        #return 2 * np.sqrt(self.get_variance() / length) * np.sqrt(kappa / length - 1) + epsilon
        pass
            
    def extend(self, new_result: float):
            self.__result_dict = np.append(self.__result_dict, np.array([new_result]))
        
    def save_data(self, filename = 'eval_results.csv'):
        try:
            np.savetxt(filename, self.__result_dict, delimiter=',')
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

            