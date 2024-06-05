import numpy as np
import random
import copy
import os
from collections import defaultdict
import pandas as pd

num_features, num_rows = 3, 20 # you can change this
dummy_data = np.random.randint(0, 100, size = (num_rows, num_features)) # generates dummy data from given parameters

# Evaluates a given subset of features (returns a random number)
def stub_evaluate(features, data):
    return np.random.uniform(0,1)

class Graph:
    def __init__(self, data=dummy_data, evaluate_function=stub_evaluate):
        print('Initializing graph object')
        if data is dummy_data:
            print('Data filepath not passed in initialization, defaulting to dummy data')
        
        if evaluate_function is stub_evaluate:
            print('Evaluate function not passed in initialization, defaulting to stub evaluate')

        self.train(data)
        self.evaluate_function = evaluate_function

        return
    
    def forward_select_features(self, old_feature_set=None, prev_score=None):
        if not old_feature_set:
            old_feature_set = set()
        best_set = old_feature_set 
        best_score = prev_score if prev_score else self.evaluate_function(old_feature_set, self.data) 
        print('Best so far:', best_set, best_score)

        # Check every potential new feature set
        for potential_feature in range(self.num_features):
            if potential_feature not in old_feature_set: 
                # Create new feature set
                new_feature_set = copy.copy(old_feature_set)
                new_feature_set.add(potential_feature)
                
                # Evaluate and compare the new feature set
                new_score = self.evaluate_function(new_feature_set, self.data)
                print('\ttrying', new_feature_set, new_score)
                if new_score > best_score:
                    best_score = new_score
                    best_set = new_feature_set
        
        # Terminate if no improvements are possible
        if best_score == prev_score:
            print('final selection:', best_set, best_score)
            return best_set, best_score
        
        # Otherwise, keep searching
        return self.forward_select_features(best_set, best_score)
        
    def backward_select_features(self, old_feature_set=None, prev_score=None):
        if not old_feature_set:
            old_feature_set = set([i for i in range(self.num_features)])
        best_set = old_feature_set 
        best_score = prev_score if prev_score else self.evaluate_function(old_feature_set, self.data)

        print('Best so far:', best_set, best_score)
        
        # Check every potential new feature set
        for potential_feature in range(self.num_features):
            if potential_feature in old_feature_set: 
                
                # Create new feature set
                new_feature_set = copy.copy(old_feature_set)
                new_feature_set.remove(potential_feature)
                
                # Evaluate and compare the new feature set
                new_score = self.evaluate_function(new_feature_set, self.data)
                print('\ttrying', new_feature_set, new_score)
                if new_score > best_score:
                    best_score = new_score
                    best_set = new_feature_set
        
        # Terminate if no improvements are possible
        if best_score == prev_score:
            print('final selection:', best_set, best_score)
            return best_set, best_score
        
        # Otherwise, keep searching
        return self.backward_select_features(best_set, best_score)

    def train(self, data): 
        print('Train called')
        
        if isinstance(data, str):
            temp_data = pd.read_csv(data, header=None, delimiter='  ')
            self.data = temp_data.iloc[:, 1:]
            self.labels = temp_data.iloc[:, 0].astype(np.int32) # data type might be an issue
        else:
            print('Non string detected')
            self.data = data
        self.num_features = self.data.shape[1]
        print(f'Number of features: {self.num_features}')

    def test(self, data_point, feature_set):
        lowest_dist, label_pred = np.inf, None
        for index, row in self.data.iterrows():
            class_label = data_point.iloc[0]
            distance = np.linalg.norm(data_point.astype(float).loc[list(feature_set)] - row.astype(float).loc[list(feature_set)])
            print(distance)
            if distance < lowest_dist:
                lowest_dist = distance
                label_pred = self.data.loc[index, 0]

        return label_pred == class_label


        


if __name__ == "__main__":
    np.random.seed(0) # sets the random seed (you can change this)
    os.system('cls' if os.name == 'nt' else 'clear') # clears the console
    
    dataset = pd.read_csv('small-test-dataset.txt', header=None, sep='  | ', engine='python')
    # print(row)
    x = Graph(data=dataset) # creates a graph instance

    # print('FORWARD')
    # x.forward_select_features() # forward selection

    # print('\nBACKWARD')
    # x.backward_select_features() # backwards elimination

    # print(dataset.loc[0])
    dummy_data_point = dataset.loc[0]
    
    print(x.test(dummy_data_point, {1, 2, 3}))
    

"""
TODO
- generate fake labels for dummy data
"""