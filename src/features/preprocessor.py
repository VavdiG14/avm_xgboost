import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self, use_robust_scaler: bool = True):
        """
        Initialize preprocessor
        
        Args:
            use_robust_scaler: If True, use RobustScaler, else StandardScaler
        """
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.label_encoders = {}
        self.scaled_features = None
        
    def _identify_features(self, df):
        """Identify numeric and categorical columns"""
        numeric_features = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        return numeric_features, categorical_features
        
    def fit_transform(self, df: pd.DataFrame, exclude_list: list = None) -> pd.DataFrame:
        """
        Fit and transform the data, handling both numeric and categorical features
        
        Args:
            df: Input DataFrame
            exclude_list: List of columns to exclude from any processing
        """
        exclude_list = exclude_list or []
        df_processed = df.copy()
        
        # Identify feature types
        numeric_features, categorical_features = self._identify_features(df)
        
        # Handle numeric features
        if len(numeric_features) > 0:
            # Determine which numeric features to scale
            cols_to_scale = [col for col in numeric_features if col not in exclude_list]
            self.scaled_features = cols_to_scale
            
            if cols_to_scale:
                df_processed[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        # Handle categorical features
        if len(categorical_features) > 0:
            # Determine which categorical features to encode
            cols_to_encode = [col for col in categorical_features if col not in exclude_list]
            
            for col in cols_to_encode:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        
        Args:
            df: Input DataFrame
        """
        df_processed = df.copy()
        
        # Transform scaled numeric features
        if self.scaled_features:
            df_processed[self.scaled_features] = self.scaler.transform(df[self.scaled_features])
        
        # Transform encoded categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df_processed[col] = encoder.transform(df[col])
        
        return df_processed 