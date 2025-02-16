from src.models.xgboost_analyzer import XGBoostAnalyzer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import numpy as np

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        self.analyzer = XGBoostAnalyzer(config)
        
    def select_by_importance(self, X_train, y_train, X_test):
        """Select features using XGBoost feature importance"""
        # Train XGBoost to get feature importance
        self.analyzer.train(X_train, y_train)
        
        # Get top N features
        selected_features = self.analyzer.feature_importance['feature'].head(self.config.n_features).tolist()
        
        # Log feature importance plot
        self.analyzer.plot_feature_importance()
        
        return selected_features
    
    def select_sequential(self, X_train, y_train, X_test, time_col='year'):
        """
        Select features using Sequential Feature Selection with time-based CV
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Test features
        time_col : str
            Name of the time column for CV splits
        
        Returns:
        --------
        list : Selected feature names
        """
        # Setup time-based cross-validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_splits,
            test_size=self.config.cv_test_size
        )
        
        # Initialize XGBoost model for feature selection
        base_model = self.analyzer.get_base_model()
        
        # Initialize Sequential Feature Selector
        sfs = SequentialFeatureSelector(
            estimator=base_model,
            n_features_to_select=self.config.n_features,
            direction='forward',
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )
        
        # Sort data by time for proper CV splits
        sorted_indices = X_train[time_col].argsort()
        X_train_sorted = X_train.iloc[sorted_indices]
        y_train_sorted = y_train.iloc[sorted_indices]
        
        # Fit SFS
        sfs.fit(X_train_sorted, y_train_sorted)
        
        # Get selected feature names
        selected_features = X_train.columns[sfs.get_support()].tolist()
        
        # Log selected features and CV scores
        #mlflow.log_param("selected_features_sequential", selected_features)
        #mlflow.log_metric("cv_score_mean", -np.mean(sfs.cv_scores_))
        #mlflow.log_metric("cv_score_std", np.std(sfs.cv_scores_))
        
        return selected_features