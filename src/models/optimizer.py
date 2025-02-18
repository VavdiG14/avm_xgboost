import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
import joblib
from pathlib import Path

class OptunaOptimizer:
    def __init__(self, config, X_train, y_train, train_years, selected_features=None):
        """
        Initialize optimizer
        
        Args:
            config: OptunaConfig object
            X_train: Training features
            y_train: Training target
            train_years: Years for time-based CV
            selected_features: List of features to use (optional)
        """
        self.config = config
        # If selected features provided, use only those
        if selected_features is not None:
            self.X_train = X_train[selected_features]
        else:
            # Try to load from saved file
            try:
                selected_features = joblib.load(PathConfig().interim_dir / "selected_features.joblib")
                self.X_train = X_train[selected_features]
            except:
                self.X_train = X_train
                
        self.y_train = y_train
        self.train_years = train_years
        self.unique_years = sorted(train_years.unique())

    def time_based_cv_score(self, model):
        """Perform time-based cross validation"""
        cv_scores = []
        
        # For each year except the last one
        for i in range(2, len(self.unique_years) + 1):
            # Get years for this fold
            years_train = self.unique_years[:i]
            year_val = years_train[-1]
            
            # Split data
            train_mask = self.train_years.isin(years_train[:-1])
            val_mask = self.train_years == year_val
            
            # Fit on all years except the last
            model.fit(
                self.X_train[train_mask], 
                self.y_train[train_mask]
            )
            
            # Predict on the last year
            val_pred = model.predict(self.X_train[val_mask])
            val_true = self.y_train[val_mask]
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(val_true, val_pred))
            cv_scores.append(rmse)
            
            # Log each fold's performance
            mlflow.log_metric(f"rmse_val_{year_val}", rmse)
        
        return np.mean(cv_scores)

    def objective(self, trial):
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }
        
        model = xgb.XGBRegressor(**params, random_state=self.config.random_state)
        cv_score = self.time_based_cv_score(model)
        
        return cv_score  # Already RMSE, no need to negate

    def optimize(self):
        """Optimize hyperparameters using Optuna"""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.config.n_trials)
        
        best_params = study.best_params
        
        # Log best parameters to MLflow
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_rmse", study.best_value)
        
        return best_params 