from personne import Personne
import pandas as pd
from CART import CART

class RandomForest:
    def __init__(self, data, n_trees, n_features):
        self.__data = data
        self.__n_trees = n_trees
        self.__n_features = n_features
        self.__trees = []
        self.__features = None
        self.__target = None
        self.__categorical = None
        self.__numerical = None

    def __bootstrap(self, data):
        return data.sample(n = data.shape[0], replace = True)
    
    def __build_tree(self, data):
        tree = CART(data)
        tree.build_tree()
        return tree
    
    def fit(self):
        for _ in range(self.__n_trees):
            data = self.__bootstrap(self.__data)
            tree = self.__build_tree(data)
            self.__trees.append(tree)
    
    def predict(self, row):
        predictions = []
        for tree in self.__trees:
            prediction = tree.predict(row)
            predictions.append(prediction)
        return pd.Series(predictions).mode().values[0]
    
    def score(self, data):
        predictions = data.apply(self.predict, axis = 1)
        return (predictions == data[self.__target]).mean()
    
    def set_features(self, features):
        self.__features = features
        self.__target = features[-1]
        self.__categorical = [feature for feature in features if self.__data[feature].dtype == 'object']
        self.__numerical = [feature for feature in features if self.__data[feature].dtype != 'object']