import pandas as pd
import numpy as np
from src.features.preprocessor import Preprocessor


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.preprocessor = Preprocessor()

    def load_data(self, path, exclude_from_scaling=None):
        """Load data and split into train/test by year"""
        path_str = str(path)
        
        if path_str.startswith("azureml://"):
            pass
        
        # Load Excel file
        df = pd.read_excel(path)
        
        # Process date and create year column
        df[self.config.year_column] = pd.to_datetime(df[self.config.date_column]).dt.year
        df["month"] = pd.to_datetime(df[self.config.date_column]).dt.month
        
        # Columns to drop: date, IDs, and any other specified columns
        columns_to_drop = [self.config.date_column] + self.config.id_columns
        df = df.drop(columns=columns_to_drop)
         
        # Split test/train by year
        test_mask = df[self.config.year_column] == self.config.test_year
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        # Split features and target
        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column]
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]
        
        # Preprocess features
        X_train_processed = self.preprocessor.fit_transform(X_train, exclude_list=exclude_from_scaling)
        X_test_processed = self.preprocessor.transform(X_test)
        
        return X_train_processed,X_test_processed, y_train, y_test, train_df[self.config.year_column] 