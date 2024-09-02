import pandas as pd

class eval_results:
    
    #TODO: number done episodes, property name
    #TODO: getter for all, mean, variance, std, confidence interval
    #TODO: add (extend)
    #TODO: save_data(filename:property_data), write all the single outcomes
    
    def __init__(result_dict):
        self.result_dict = result_dict
    
    def get_result(property_name: str):
        try:
            result = self.result_dict[property_name]
            return result
        except:
            print("metric_name must be string")
        

            