import pandas as pd
from personne import Personne

class CART():
    def __init__(self, data):
        self.__data = data
        self.__root = None
        self.__tree = None
        self.__features = None
        self.__target = None
        self.__categorical = None
        self.__numerical = None

    def __split(self, data, feature, value):
        if feature in self.__categorical:
            left = data[data[feature] == value]
            right = data[data[feature] != value]
        else:
            left = data[data[feature] <= value]
            right = data[data[feature] > value]
        return left, right
    
    def __gini(self, data):
        if data.empty:
            return 0
        p = data[self.__target].value_counts(normalize = True)
        return 1 - (p ** 2).sum()
    
    def __best_split(self, data):
        best_feature = None
        best_value = None
        best_gini = 1
        for feature in self.__features:
            values = data[feature].unique()
            for value in values:
                left, right = self.__split(data, feature, value)
                gini = (left.shape[0] * self.__gini(left) + right.shape[0] * self.__gini(right)) / data.shape[0]
                if gini < best_gini:
                    best_feature = feature
                    best_value = value
                    best_gini = gini
        return best_feature, best_value, best_gini
    
    def __build_tree(self, data):
        feature, value, gini = self.__best_split(data)
        if gini == 0:
            return data[self.__target].mode().values[0]
        left, right = self.__split(data, feature, value)
        left_tree = self.__build_tree(left)
        right_tree = self.__build_tree(right)
        return (feature, value, left_tree, right_tree)
    
    def __predict(self, tree, row):
        if tree in self.__target:
            return tree
        feature, value, left_tree, right_tree = tree
        if feature in self.__categorical:
            if row[feature] == value:
                return self.__predict(left_tree, row)
            else:
                return self.__predict(right_tree, row)
        else:
            if row[feature] <= value:
                return self.__predict(left_tree, row)
            else:
                return self.__predict(right_tree, row)
    
    def fit(self, target, categorical, numerical):
        self.__target = target
        self.__categorical = categorical
        self.__numerical = numerical
        self.__features = categorical + numerical
        self.__root = self.__build_tree(self.__data)
    
    def predict(self, data):
        self.__tree = data.apply(lambda row: self.__predict(self.__root, row), axis = 1)
        return self.__tree
    
    def evaluate(self):
        return (self.__tree == self.__data[self.__target]).mean()
    
    def build_tree(self):
        self.__tree = self.__build_tree(self.__data)
        return self.__tree 