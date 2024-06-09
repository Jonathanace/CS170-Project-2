import numpy as np
import random
import copy
import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm, trange

num_features, num_rows = 3, 20 # you can change this
dummy_data = np.random.randint(0, 100, size = (num_rows, num_features)) # generates dummy data from given parameters

# Evaluates a given subset of features (returns a random number)
def stub_evaluate(features, data):
    return np.random.uniform(0,1)

class Graph:
    def __init__(self, data=dummy_data):
        # print('Initializing graph object')
        if data is dummy_data:
            print('Data filepath not passed in initialization, defaulting to dummy data')
        
        # print(data[0])
        self.train(data)

        return
    
    def forward_select_features(self, old_feature_set=None, prev_score=0):
        if not old_feature_set:
            old_feature_set = set()
        best_set = old_feature_set 
        best_score = self.evaluate(old_feature_set)
        print('Best so far:', best_set, best_score)

        # Check every potential new feature set
        for potential_feature in range(1, self.num_features):
            if potential_feature not in old_feature_set: 
                # Create new feature set
                new_feature_set = copy.copy(old_feature_set)
                new_feature_set.add(potential_feature)
                
                # Evaluate and compare the new feature set
                new_score = self.evaluate(new_feature_set)
                print('\ttrying', new_feature_set, new_score)
                if new_score >= best_score:
                    best_score = new_score
                    best_set = new_feature_set
        

        # print(f'{best_set}, {best_score}, {prev_score}')
        # Terminate if no improvements are possible
        if best_score == prev_score:
            print('final selection:', best_set, best_score)
            return best_set, best_score
        
        # Otherwise, keep searching
        return self.forward_select_features(best_set, best_score)
        
    def backward_select_features(self, old_feature_set=None, prev_score=None):
        if not old_feature_set:
            old_feature_set = set([i for i in range(1, self.num_features)])
        if not prev_score:
            prev_score = self.evaluate(old_feature_set)
        
        best_set = old_feature_set 
        best_score = prev_score
        print('Best so far:', best_set, best_score)
        
        # Check every potential new feature set
        for potential_feature in range(1, self.num_features):
            if potential_feature in old_feature_set: 
                
                # Create new feature set
                new_feature_set = copy.copy(old_feature_set)
                new_feature_set.remove(potential_feature)
                
                # Evaluate and compare the new feature set
                new_score = self.evaluate(new_feature_set)
                print('\ttrying', new_feature_set, new_score)
                if new_score >= best_score:
                    best_score = new_score
                    best_set = new_feature_set
        
        # Terminate if no improvements are possible
        if best_score == prev_score:
            print('final selection:', best_set, best_score)
            return best_set, best_score
        
        # Otherwise, keep searching
        return self.backward_select_features(best_set, best_score)

    def train(self, data): 
        # print('Train called')
        self.data = data
        self.num_features = self.data.shape[1]
        # print(f'Number of features: {self.num_features}')

    def test(self, test_index, feature_set):
        test_dataset = np.delete(self.data, test_index, 0)
        test_point = self.data[test_index, list(feature_set)]
        correct_label = self.data[test_index, 0]
        # print(list(feature_set))
        test_dataset_features = test_dataset[:, list(feature_set)]

        # print(test_dataset_features.shape,  test_point.shape, correct_label)
        distance_dataset = np.linalg.norm(test_dataset_features - test_point, axis=1)
        pred_index = np.argmin(distance_dataset)
        pred_label = test_dataset[pred_index, 0]
        
        # print(pred_label, correct_label)
        return pred_label == correct_label

    def evaluate(self, feature_set=None):
        i, j = 0, 0
        n = len(self.data)
        for test_index in range(n):
            # print(f'evaluating {test_index}')
            if self.test(test_index, feature_set):
                i += 1
            else:
                j += 1
        score = i/(i+j)
        # print(f'{feature_set}: {score}')
        return score
            


    # def evaluate(self, points, feature_set):
    #     i, j = 0, 0
    #     self.train(points)
    #     for index, data_point in points.iterrows():
    #         print(f'evaluating row {index}')
    #         if self.test(data_point, feature_set, leave_out=True):
    #             i += 1
    #         else:
    #             j += 1
    #     return i / (i + j)
        


if __name__ == "__main__":
    np.random.seed(0) # sets the random seed (you can change this)
    os.system('cls' if os.name == 'nt' else 'clear') # clears the console
    
    dataset = pd.read_csv('large-test-dataset.txt', header=None, sep='  | ', engine='python').to_numpy()
    # print(dataset.shape)
    x = Graph(dataset)
    # print(x.evaluate({3,5}))
    x.forward_select_features()
    
