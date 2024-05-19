import numpy as np
import random
import copy
import os
from collections import defaultdict

num_features, num_rows = 3, 20 # you can change this
dummy_data = np.random.randint(0, 100, size = (num_rows, num_features)) # generates dummy data from given parameters

# Evaluates a given subset of features (returns a random number)
def stub_evaluate(features, data):
    return np.random.uniform(0,1)

class Graph:
    def __init__(self, evaluate_function=stub_evaluate, data=dummy_data):
        self.num_features = data.shape[1]
        self.evaluate_function = evaluate_function
        self.data = data
        print('Initializing graph')
        print(f'Number of features: {self.num_features}')
        # print(dummy_data)
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

if __name__ == "__main__":
    np.random.seed(0) # sets the random seed (you can change this)
    os.system('cls' if os.name == 'nt' else 'clear') # clears the console
    
    x = Graph() # creates a graph instance

    print('FORWARD')
    x.forward_select_features() # forward selection
    print('\nBACKWARD')
    x.backward_select_features() # backwards elimination