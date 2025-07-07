#!/usr/bin/env python3
"""Data Science Pipeline"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class DataSciencePipeline:
    def __init__(self):
        self.model = None
    
    def load_data(self, file_path):
        return pd.read_csv(file_path)
    
    def train_model(self, X, y):
        self.model = RandomForestClassifier()
        self.model.fit(X, y)
        return self.model

if __name__ == "__main__":
    print("Data Science Pipeline initialized")
